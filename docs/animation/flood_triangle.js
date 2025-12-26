let allPoints = [];      // from points.csv (all points)
let landmarks = [];      // from landmarks.csv (subset points)
let edges = [];          // edges between landmarks
let radiusSlider;
let animating = true;
const animationSpeed = 0.006;

function preload() {
  tablePoints = loadTable('animation/points.csv', 'csv', 'noHeader');
  tableLandmarks = loadTable('animation/landmarks.csv', 'csv', 'noHeader');
  tableEdges = loadTable('animation/edges.csv', 'csv', 'noHeader');
  tableTriangles = loadTable('animation/triangles.csv', 'csv', 'noHeader');
}

function setup() {
  const canvas = createCanvas(600, 300);
  canvas.parent("canvas-container");

  const currentScheme = document.documentElement.getAttribute('data-md-color-scheme');
  const isDark = currentScheme === 'slate';

  // Define color palette for light/dark mode
  colors = isDark ? {
    background: 0,
    pointBall: [5, 67, 255],    // orange
    pointDot: 255,
    landmark: [87, 87, 235],
    edgeActive: [135, 206, 250, 180], // lightblue
    edgeInactive: [255, 255, 255, 100],
    triangle: [87, 87, 235, 50]
  } : {
    background: 255,
    pointBall: [160, 170, 180], // [5, 67, 255],    // blue
    pointDot: 0,
    landmark: [255, 193, 5],
    edgeActive: [255, 193, 5, 255],
    edgeInactive: [255, 193, 5],
    triangle : [255, 193, 5, 100]//:[255, 213.7, 88.3]
  };

  // Get buttons
  const playPauseBtn = select('#playPauseBtn');
  const icon = playPauseBtn.elt.querySelector('i');

  // Set initial icon to pause since animating is true
  icon.classList.remove('fa-play');
  icon.classList.add('fa-pause');
  

  playPauseBtn.mousePressed(() => {
    const icon = playPauseBtn.elt.querySelector('i');
    if (!animating && parseFloat(radiusSlider.value()) >= parseFloat(radiusSlider.attribute('max'))) {
      radiusSlider.value(0);
    }
  
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
  radiusSlider.attribute("max", 2.5);
  radiusSlider.attribute("step", "0.01");
  radiusSlider.value(0);
}

function draw() {
  clear();
  if (animating) {
    let val = parseFloat(radiusSlider.value());
    val += animationSpeed;
    if (val >= parseFloat(radiusSlider.attribute('max'))) {
      val = parseFloat(radiusSlider.attribute('max'));
      radiusSlider.value(val);
      animating = false;

      // Change button icon to "Play" when animation ends
      const icon = document.querySelector('#playPauseBtn i');
      icon.classList.remove('fa-pause');
      icon.classList.add('fa-play');
    } else {
      radiusSlider.value(val);
    }
  }
  // const t = parseFloat(radiusSlider.value());
  const initialOffset = 0.275;  // or whatever value you want to start with

  let t = parseFloat(radiusSlider.value()) + initialOffset;

  const maxDataRadius = 0.05;
  const dataRadius = maxDataRadius * t;
  const pixelRadius = dataRadius * (width - 2 * padding);

  // Draw filled gray balls around all points
  fill(...colors.pointBall);
  // fill(200);
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
      fill(...colors.triangle);
      noStroke();
    } else {
      noFill();
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
      stroke(...colors.edgeActive);
      strokeWeight(3);
      drawingContext.setLineDash([]);
    } else {
      stroke(...colors.edgeInactive);
      strokeWeight(1);
      drawingContext.setLineDash([2,4]);
    }
    line(...p1, ...p2);
  }
  drawingContext.setLineDash([]);

  // Draw all points
  fill(colors.pointDot);
  noStroke();
  for (const p of allPoints) {
    const [x, y] = toCanvasCoords(p);
    circle(x, y, 6);
  }

  // Draw landmarks
  fill(...colors.landmark);
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