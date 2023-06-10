# CT_FiberSegentation
segment the short fibes filled in the polymer matrix from the CT 3D volume data
# The processing work flow is as below: 
```mermaid
flowchart TD
D([Origin Data]) -- correlation operation --> O([orientation field]) -->T
D --> B(binarized mask) --> S(skeleton) --> T(traced fibers)
```
# why
<!-- toc -->

- [CT\_FiberSegentation](#ct_fibersegentation)
- [The processing work flow is as below:](#the-processing-work-flow-is-as-below)
- [why](#why)
    - [origin data](#origin-data)
    - [local view](#local-view)
    - [final lines](#final-lines)
  
<!-- tocstop -->
### origin data
![Origin Data](./demo_data/origin_data.png)
### local view
![local view](./demo_data/local_view.png)
### final lines
![final lines](./demo_data/final lines.png)