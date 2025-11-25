

```mermaid
graph TD
    File[EPW / CSV File] -->|ScenarioLoader (CPU)| GT[Ground Truth ExogenousData]
    GT -->|Upload| VRAM[GPU VRAM]
    
    subgraph GPU Simulation Loop [Step t]
        VRAM -->|Slice [t]| RealData[Current Exo]
        VRAM -->|Slice [t:t+H]| FutureData[Perfect Horizon]
        
        RealData -->|PhysicsEngine| RealCtx[Real Context]
        RealCtx -->|DAE Solver| NextState
        
        FutureData -->|Forecaster| NoisyForecast[Forecast Exo]
        NoisyForecast -->|MPC Solver| Action
    end
```