import React, { useMemo, useState } from "react";

/**
 * KolamQR — text→kolam code (QR‑inspired, high capacity)
 * ------------------------------------------------------
 * Goal: pack LOTS of bits like a QR, but render each data module
 * as a small sikku‑style curve motif (novel + authentic vibe).
 *
 * Design:
 * - Only input = text.
 * - Auto chooses module grid size MxM (odd), like QR versions (25, 29, 33, ...).
 * - Reserves three finder diamonds + timing lines.
 * - Data mapped by zig‑zag scan (QR‑style), skipping reserved areas.
 * - Each data module uses one of 4 kolam motifs (2 bits per module).
 *   Motifs are 2×2 dot tiles that keep degree‑2 locally.
 * - Output is SVG.
 *
 * This is NOT a QR reader/writer; it's a “kolam code” aesthetic encoder
 * with high capacity and familiar structure.
 */

// ---- Utils --------------------------------------------------------------
const textToBits = (txt) => {
  const bytes = new TextEncoder().encode(txt);
  const bits = [];
  for (const b of bytes) for (let i = 7; i >= 0; i--) bits.push((b >> i) & 1);
  return bits;
};

const hash32 = (s) => {
  let h = 2166136261 >>> 0; // FNV-ish
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
};

class RNG { constructor(seed){this.s=seed>>>0;} next(){let x=this.s; x^=x<<13; x^=x>>>17; x^=x<<5; this.s=x>>>0; return (this.s&0xffffffff)/0x100000000;} bit(){return this.next()<0.5?0:1;} }

// Choose module size to fit bits at 2 bits/module minus fixed patterns
const chooseModuleSize = (bitCount) => {
  // QR-like version sizes (odd): 25, 29, 33, 37, 41, ...
  const versions = Array.from({length:20}, (_,k)=>25+4*k);
  for (const M of versions){
    const dataModules = (M*M) - finderFootprint(M) - timingFootprint(M);
    if (Math.floor(dataModules*2) >= bitCount) return M;
  }
  return versions[versions.length-1];
};

const finderFootprint = (M)=> 3*7*7; // three 7x7 finder diamonds
const timingFootprint = (M)=> (M-14)*2; // one timing row + one timing col excluding overlaps

// Zigzag traversal of data modules (skip reserved)
const isReserved = (M, r, c) => {
  // Finder squares at TL, TR, BL (7x7)
  const inFinder = (rr,cc, R,C)=> rr>=R && rr<R+7 && cc>=C && cc<C+7;
  if (inFinder(r,c,0,0) || inFinder(r,c,0,M-7) || inFinder(r,c,M-7,0)) return true;
  // Timing: row 6 and col 6 (0-index), excluding finder areas
  if (r===6 || c===6) return true;
  return false;
};

const zigzagCoords = (M) => {
  const coords=[];
  let dir=-1; // start upward
  for (let c=M-1; c>=0; c-=2){
    if (c===6) c--; // skip timing column 6
    const colPair=[c, c-1];
    for (const cc of colPair){
      for (let r= (dir>0?0:M-1); dir>0? r<M : r>=0; r+=dir){
        if (cc===6 || r===6) continue; // skip timing
        if (isReserved(M,r,cc)) continue;
        coords.push([r,cc]);
      }
    }
    dir*=-1; // flip direction
  }
  return coords;
};

// ---- Motifs (2 bits per module) ----------------------------------------
// We draw each module as a tiny sikku motif inside a cell. Motifs are designed
// to look braided and distinct while being compact.
const motifPath = (x, y, s, type) => {
  const r = s*0.38; // radius
  const cx=x+s/2, cy=y+s/2;
  const arc=(a0,a1)=>{
    const x0=cx+r*Math.cos(a0), y0=cy+r*Math.sin(a0);
    const x1=cx+r*Math.cos(a1), y1=cy+r*Math.sin(a1);
    return `M ${x0.toFixed(2)} ${y0.toFixed(2)} A ${r.toFixed(2)} ${r.toFixed(2)} 0 0 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;
  };
  switch(type){
    case 0: // cross weave
      return [arc(Math.PI,1.5*Math.PI), arc(0,0.5*Math.PI), arc(0.5*Math.PI,Math.PI), arc(1.5*Math.PI,2*Math.PI)].join(" ");
    case 1: // diagonal weave NE-SW
      return [arc(0,0.5*Math.PI), arc(Math.PI,1.5*Math.PI)].join(" ");
    case 2: // diagonal weave NW-SE
      return [arc(0.5*Math.PI,Math.PI), arc(1.5*Math.PI,2*Math.PI)].join(" ");
    case 3: // curl/rosette
      const rr=r*0.7; return `M ${(cx+rr).toFixed(2)} ${cy.toFixed(2)} a ${rr.toFixed(2)} ${rr.toFixed(2)} 0 1 1 ${(-2*rr).toFixed(2)} 0 a ${rr.toFixed(2)} ${rr.toFixed(2)} 0 1 1 ${(2*rr).toFixed(2)} 0`;
    default: return "";
  }
};

// Finder diamond motif at 7x7 with braided border
const drawFinder = (x, y, s) => {
  const size = 7*s; const cx=x+3.5*s, cy=y+3.5*s; const r=3*s; const rr=2*s; const rrr=1*s;
  const ring=(rad,stroke)=>`<circle cx='${cx}' cy='${cy}' r='${rad}' stroke='${stroke}' stroke-width='${s*0.9}' fill='none'/>`;
  return `${ring(r,'currentColor')}${ring(rr,'currentColor')}${ring(rrr,'currentColor')}`;
};

// Timing pattern motif (alternating)
const timingStroke = (x,y,w,h,flip)=>`<rect x='${x}' y='${y}' width='${w}' height='${h}' rx='${Math.min(w,h)/2}' fill='currentColor' opacity='${flip?0.5:0.9}'/>`;

// ---- Main component -----------------------------------------------------
export default function KolamQR(){
  const [text,setText]=useState("Kolam Code — make it dense like QR but pretty");

  const svg = useMemo(()=>{
    const bits = textToBits(text);
    const M = chooseModuleSize(bits.length);
    const dataCoords = zigzagCoords(M);
    // pack 2 bits per module; pad with PRNG
    const needModules = Math.ceil(bits.length/2);
    const rng = new RNG(hash32(text));
    const packed = bits.slice();
    while (Math.ceil(packed.length/2) < needModules) packed.push(rng.bit());

    // Build module map
    const mod = Array.from({length:M},()=>Array(M).fill(null));

    // place finders
    const placeFinder=(R,C)=>{ for(let r=0;r<7;r++) for(let c=0;c<7;c++) mod[R+r][C+c] = {kind:'finder'}; };
    placeFinder(0,0); placeFinder(0,M-7); placeFinder(M-7,0);

    // timing lines row 6 and col 6
    for(let i=0;i<M;i++){ if(mod[6][i]==null) mod[6][i]={kind:'timing', flip:i%2}; if(mod[i][6]==null) mod[i][6]={kind:'timing', flip:i%2}; }

    // fill data modules
    let bi=0; for (let k=0;k<dataCoords.length && bi<packed.length; k++){
      const [r,c]=dataCoords[k]; if(mod[r][c]!=null) continue;
      const b1=packed[bi++]??rng.bit(); const b2=packed[bi++]??rng.bit();
      const val = (b1<<1)|b2; mod[r][c]={kind:'data', val};
    }

    // Render SVG
    const cell = 22; const pad = 18; const W = M*cell+pad*2, H=W;
    let out = `<svg xmlns='http://www.w3.org/2000/svg' width='${W}' height='${H}' viewBox='0 0 ${W} ${H}'>`;
    out += `<rect width='100%' height='100%' fill='#121212'/>`;
    out += `<g transform='translate(${pad} ${pad})' stroke='#f6e7d7' fill='none' stroke-linecap='round'>`;

    for(let r=0;r<M;r++){
      for(let c=0;c<M;c++){
        const cellX=c*cell, cellY=r*cell;
        const t = mod[r][c]; if(!t) continue;
        if(t.kind==='finder'){
          out += `<g color='#f6e7d7'>${drawFinder(cellX,cellY,cell)}</g>`;
        } else if(t.kind==='timing'){
          out += timingStroke(cellX+cell*0.35, cellY+cell*0.35, cell*0.3, cell*0.3, t.flip);
        } else if(t.kind==='data'){
          const d = motifPath(cellX,cellY,cell,t.val);
          out += `<path d='${d}' stroke-width='4'/>`;
        }
      }
    }

    out += `</g></svg>`; return {svg: out, size: W, M};
  },[text]);

  const download = () => { const blob=new Blob([svg.svg],{type:'image/svg+xml'}); const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download='kolamqr.svg'; a.click(); URL.revokeObjectURL(url); };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-2">KolamQR — dense, QR-inspired kolam code</h1>
      <p className="text-sm mb-3 text-gray-600">Only input is text. High-capacity module grid with finder diamonds + timing lines. Each data cell is a sikku-style motif (2 bits per module).</p>
      <input className="w-full border rounded px-3 py-2 mb-3" value={text} onChange={(e)=>setText(e.target.value)} />
      <div className="text-sm mb-3">Modules: {svg.M}×{svg.M}</div>
      <div className="bg-[#1a1a1a] rounded-xl p-4 flex items-center justify-center" dangerouslySetInnerHTML={{__html: svg.svg}} />
      <button onClick={download} className="mt-4 px-4 py-2 bg-black text-white rounded">Download SVG</button>
    </div>
  );
}
