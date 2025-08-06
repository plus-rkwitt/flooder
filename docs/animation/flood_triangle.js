let allPoints = [];      // from points.csv (all points)
let landmarks = [];      // from landmarks.csv (subset points)
let edges = [];          // edges between landmarks
let radiusSlider;
let animating = true;
const animationSpeed = 0.01;

function preload() {
  tablePoints = loadTable('animation/points.csv', 'csv', 'noHeader');
  tableLandmarks = loadTable('animation/landmarks.csv', 'csv', 'noHeader');
  tableEdges = loadTable('animation/edges.csv', 'csv', 'noHeader');
  tableTriangles = loadTable('animation/triangles.csv', 'csv', 'noHeader');
}

function setup() {
  const canvas = createCanvas(600, 300);
  canvas.parent("canvas-container");
  // Get buttons
  const playPauseBtn = select('#playPauseBtn');
  const icon = playPauseBtn.elt.querySelector('i');

  // Set initial icon to pause since animating is true
  icon.classList.remove('fa-play');
  icon.classList.add('fa-pause');

  playPauseBtn.mousePressed(() => {
    animating = !animating;
    if (animating) {
      icon.classList.remove('fa-play');
      icon.classList.add('fa-pause');
    } else {
      icon.classList.remove('fa-pause');
      icon.classList.add('fa-play');
    }
  });
  
  // Load all points
  allPoints = [];
  for (let r = 0; r < tablePoints.getRowCount(); r++) {
    let x = parseFloat(tablePoints.getString(r, 0));
    let y = parseFloat(tablePoints.getString(r, 1));
    allPoints.push([x, y]);
  }

  // Load landmarks (subset points)
  landmarks = [];
  for (let r = 0; r < tableLandmarks.getRowCount(); r++) {
    let x = parseFloat(tableLandmarks.getString(r, 0));
    let y = parseFloat(tableLandmarks.getString(r, 1));
    landmarks.push([x, y]);
  }

  // Load edges connecting landmarks (indices into landmarks)
  edges = [];
  for (let r = 0; r < tableEdges.getRowCount(); r++) {
    const i = int(tableEdges.getString(r, 0));
    const j = int(tableEdges.getString(r, 1));
    const radius = float(tableEdges.getString(r, 2));
    edges.push({ i, j, radius });
  }

  triangles = [];
  for (let r = 0; r < tableTriangles.getRowCount(); r++) {
    const i = int(tableTriangles.getString(r, 0));
    const j = int(tableTriangles.getString(r, 1));
    const k = int(tableTriangles.getString(r, 2));
    const radius = float(tableTriangles.getString(r, 3));
    triangles.push([i, j, k, radius]);
  }

  // Slider setup
  radiusSlider = select("#radiusSlider");
  radiusSlider.attribute("max", 4);
  radiusSlider.attribute("step", "0.01");
  radiusSlider.value(0);
}

function draw() {
  background(255);
  if (animating) {
    let val = parseFloat(radiusSlider.value());
    val += animationSpeed;  // animationSpeed should be a small float, e.g., 0.01
    if (val > parseFloat(radiusSlider.attribute('max'))) {
      val = parseFloat(radiusSlider.attribute('max')); // stop at max
      animating = false; // stop animation automatically
    }
    
    radiusSlider.value(val);
  }
  
  const t = parseFloat(radiusSlider.value());


  const maxDataRadius = 0.05; // max radius in data coords (0..1)
  const dataRadius = maxDataRadius * t;  // current radius in data units

  // Convert data radius to pixels:
  const pixelRadius = dataRadius * (width - 2 * padding);

  // Draw filled gray balls around all points
  fill(200);
  noStroke();
  for (const p of allPoints) {
    const [x, y] = toCanvasCoords(p);
    circle(x, y, 2 * pixelRadius);
  }

  // Draw triangles
  for (const [i, j, k, radius] of triangles) {
    const p1 = toCanvasCoords(landmarks[i]);
    const p2 = toCanvasCoords(landmarks[j]);
    const p3 = toCanvasCoords(landmarks[k]);
    if (dataRadius >= radius) {
      fill(87, 87, 235, 50);  // light gray, transparent fill
      noStroke();
    } else {
      noFill();  // light gray, transparent fill
      noStroke();
    }
    triangle(...p1, ...p2, ...p3);
  }

  // Draw edges between landmarks
  for (const { i, j, radius } of edges) {
    const p1 = toCanvasCoords(landmarks[i]);
    const p2 = toCanvasCoords(landmarks[j]);

    // radius here is in data units, so compare directly
    if (dataRadius >= radius) {
      stroke(87, 87, 235, 150);
      strokeWeight(3);
      drawingContext.setLineDash([]);
    } else {
      stroke(0, 100);
      strokeWeight(1);
      drawingContext.setLineDash([2,4]);
    }
    line(...p1, ...p2);
  }
  drawingContext.setLineDash([]);

  // Draw all points as small gray dots
  fill(0);
  noStroke();
  for (const p of allPoints) {
    const [x, y] = toCanvasCoords(p);
    circle(x, y, 6);
  }

  // Draw landmarks as black circles on top
  fill(87, 87, 235);
  noStroke();
  for (const p of landmarks) {
    const [x, y] = toCanvasCoords(p);
    circle(x, y, 10);
  }
}

const padding = 150; // pixels padding around edges

function toCanvasCoords([x, y]) {
  return [
    padding + x * (width - 2 * padding),
    padding + (.5 - y) * (width - 2 * padding)
  ];
}