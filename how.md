```mermaid

    graph TD
    A[START] --> B{1. DEFINE CANVAS};
    B --> B1[Grid Type G?];
    B1 --> B2[Dimensions D?];
    B2 --> B3[Symmetry S?];
    B3 --> C{2. DEFINE OPERATOR};
    C --> C1{Style: Pulli or Sikku?};
    C1 -- Pulli --> C2[Define Dot Sequence];
    C1 -- Sikku --> C3[Define Turn Rules f(pos, dir, grid)];
    C2 --> D;
    C3 --> D;
    D{3. DEFINE CONSTRAINT};
    D --> D1[Connectivity K? (e.g., Single Loop)];
    D1 --> E{4. EXECUTE & VALIDATE};
    E --> E1[Initialize Start Point];
    E1 --> E2[Apply Operator 'O' to draw next segment];
    E2 --> E3{Enforce Symmetry 'S' at each step};
    E3 --> E4{End of Path?};
    E4 -- No --> E2;
    E4 -- Yes --> F{5. VALIDATION};
    F --> F1{Does pattern satisfy Constraint 'K'?};
    F1 -- Yes --> G[END: Valid Kolam Generated];
    F1 -- No --> H[FAIL: Invalid Parameters, Backtrack or Restart];
    
```

