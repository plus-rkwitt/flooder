---
hide:
  - navigation
---

# Visualizations

We visualize the Flood complex at different radii on selected point clouds. All input point clouds consist of 1 million points and we always build the complex using 1000 landmarks.

<style>
  .pointcloud-canvas {
    border: 2px solid black;   /* black border */
    display: block;            /* ensures margin works */
    margin: 0 auto;            /* center horizontally */
  }

  .button-container {
    display: flex;
    justify-content: center;
    gap: 1em;
    margin: 1em 0 2em 0;       /* top 1em, bottom 2em for extra vspace */
  }
</style>

### Virus (RV-A89)
A point cloud extracted from [cryo-electron microscopy map](https://www.emdataresource.org/EMD-50844) of a rhinovirus 
<div id="canvas-virus" class="pointcloud-canvas" style="width:600px; height:300px; margin:0 auto; padding:0;"></div>
<div class="button-container">
  <button id="prevBtn_virus">◀ Back</button>
  <button id="nextBtn_virus">Next ▶</button>
</div>

### Coral
An Acropora cervicornis coral from the [Smithsonian 3D Digitization program](https://3d.si.edu/object/3d/acropora-cervicornis:8e3e67d1-591a-4488-b437-dee35e796d9e). 
<div id="canvas-coral" class="pointcloud-canvas" style="width:600px; height:300px; margin:0 auto; padding:0;"></div>
<div class="button-container">
  <button id="prevBtn_coral">◀ Back</button>
  <button id="nextBtn_coral">Next ▶</button>
</div>

### Lockwasher
A lockwasher from the [mcb dataset](https://link.springer.com/chapter/10.1007/978-3-030-58523-5_11).
<div id="canvas-lockwasher" class="pointcloud-canvas" style="width:600px; height:300px; margin:0 auto; padding:0;"></div>
<div class="button-container">
  <button id="prevBtn_lockwasher">◀ Back</button>
  <button id="nextBtn_lockwasher">Next ▶</button>
</div>

<script src="https://cdn.jsdelivr.net/npm/p5@1.9.0/lib/p5.min.js"></script>
<script src="../visualization/visualization_virus.js"></script>

<script src="https://cdn.jsdelivr.net/npm/p5@1.9.0/lib/p5.min.js"></script>
<script src="../visualization/visualization_coral.js"></script>

<script src="https://cdn.jsdelivr.net/npm/p5@1.9.0/lib/p5.min.js"></script>
<script src="../visualization/visualization_lockwasher.js"></script>
