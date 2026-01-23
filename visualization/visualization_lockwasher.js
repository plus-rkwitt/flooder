new p5(p => {

  // -----------------------------
  // CHOOSE DATASET HERE:
  // -----------------------------
  const DATASET = "lockwasher";  // "coral", "lockwasher" or "virus"
  // -----------------------------

  let points = [], edges = [], triangles = [], tetrahedra = [];
  let centerX = 0, centerY = 0, centerZ = 0;
  const snapshots = 10;
  let tIndex = 0;

  // User interaction state
  let rotX_user = 0;
  let rotY_user = 0;
  let zoom = 1;           // scroll wheel zoom
  let isDragging = false;
  let lastMouseX = 0;
  let lastMouseY = 0;

  // Settings per dataset
  const DATA = {
    lockwasher: {
      scale: 400,
      tValues: [0.,0.001,0.00125,0.0015,0.002,0.003,0.005,0.01,.1,1.],
      folder: "../visualization/lockwasher/"
    },
    coral: {
      scale: 2500,
      tValues: [0.,0.0005,0.00075,0.001,0.00125,0.0015,0.0025,0.005,0.01,.1,1.],
      folder: "../visualization/coral/"
    },
    virus: {
      scale: 0.3,
      tValues: [0.,2,3,4,5,7.5,10,15,30],
      folder: "../visualization/virus/"
    }
  };

  const {scale: scaleFactor, tValues, folder} = DATA[DATASET];

  let triangleMesh = [], tetraMesh = [], edgeMesh = [];

  let colors = {
    background: [255, 255, 255],
    point: [0, 0, 0],
    edgeActive: [255, 193, 5],
    triangle: [255, 193, 5, 150],
    tetra: [255, 193, 5, 100]
  };

  let tableLandmarks, tableEdges, tableTriangles, tableTetra;

  // -----------------------------
  // PRELOAD
  // -----------------------------
  p.preload = () => {
    tableLandmarks = p.loadTable(folder + "landmarks.csv", "csv", "noHeader");
    tableEdges     = p.loadTable(folder + "edges.csv", "csv", "noHeader");
    tableTriangles = p.loadTable(folder + "triangles.csv", "csv", "noHeader");
    tableTetra     = p.loadTable(folder + "tetrahedra.csv", "csv", "noHeader");
  };

  // -----------------------------
  // SETUP
  // -----------------------------
  p.setup = () => {
    const container = document.getElementById('canvas-lockwasher');
    if (!container) return;

    const height = container.clientHeight || 300;
    const width = height * 2;

    p.createCanvas(width, height, p.WEBGL).parent(container);
    p.pixelDensity(2);

    // Load + scale points
    points = tableLandmarks.getArray()
              .map(r => r.map(Number))
              .map(([x, y, z]) => [x * scaleFactor, y * scaleFactor, z * scaleFactor]);

    edges      = tableEdges.getArray().map(r => ({i:+r[0], j:+r[1], radius:+r[2]}));
    triangles  = tableTriangles.getArray().map(r => ({i:+r[0], j:+r[1], k:+r[2], radius:+r[3]}));
    tetrahedra = tableTetra.getArray().map(r => ({i:+r[0], j:+r[1], k:+r[2], l:+r[3], radius:+r[4]}));

    // Compute bounding-box center
    let minX=Infinity, minY=Infinity, minZ=Infinity;
    let maxX=-Infinity, maxY=-Infinity, maxZ=-Infinity;
    for (const [x,y,z] of points) {
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }
    centerX = (minX + maxX) / 2;
    centerY = (minY + maxY) / 2;
    centerZ = (minZ + maxZ) / 2;

    triangleMesh = buildTriangleMesh(triangles, points);
    tetraMesh    = buildTetraMesh(tetrahedra, points);
    edgeMesh     = buildEdgeMesh(edges, points);

    p.noLoop();
    drawSnapshot();
  };

  // -----------------------------
  // MOUSE + SCROLL INTERACTION
  // -----------------------------

  p.mousePressed = () => {
    // Only start drag if mouse is over this canvas
    if (p.mouseX >= 0 && p.mouseX <= p.width &&
        p.mouseY >= 0 && p.mouseY <= p.height) {
      isDragging = true;
      lastMouseX = p.mouseX;
      lastMouseY = p.mouseY;
    }
  };
  
  p.mouseDragged = () => {
    if (!isDragging) return;
    const dx = p.mouseX - lastMouseX;
    const dy = p.mouseY - lastMouseY;
    rotY_user += dx * 0.01;
    rotX_user += dy * 0.01;
    lastMouseX = p.mouseX;
    lastMouseY = p.mouseY;
    drawSnapshot();
  };
  
  p.mouseReleased = () => { isDragging = false; };
  
  p.mouseDragged = () => {
    if (!isDragging) return;
  
    const dx = p.mouseX - lastMouseX;
    const dy = p.mouseY - lastMouseY;
  
    rotY_user += dx * 0.01;
    rotX_user += dy * 0.01;
  
    lastMouseX = p.mouseX;
    lastMouseY = p.mouseY;
  
    drawSnapshot();
  };
  
  // Scroll wheel zoom
  p.mouseWheel = (event) => {
    if (p.mouseX >= 0 && p.mouseX <= p.width &&
        p.mouseY >= 0 && p.mouseY <= p.height) {
      zoom *= event.delta < 0 ? 1.1 : 0.9;
      zoom = Math.max(0.1, Math.min(10, zoom));
      drawSnapshot();
      return false;  // prevent page scroll
    }
    // Otherwise, allow normal page scroll
    return true;
  };

  // -----------------------------
  // DRAW SNAPSHOT
  // -----------------------------
  function drawSnapshot() {
    p.background(...colors.background);
    p.push();

    // Base fixed orientation
    p.rotateX(p.HALF_PI + 0.1);
    p.rotateY(p.PI);

    // User rotation
    p.rotateX(rotX_user);
    p.rotateY(rotY_user);

    // Zoom (scale)
    p.scale(zoom);

    // Center model
    p.translate(-centerX, -centerY, -centerZ);

    const t = tValues[tIndex];

    // Draw points
    p.push();
    p.stroke(...colors.point);
    p.strokeWeight(4);
    for (const pt of points) p.point(...pt);
    p.pop();

    // Draw edges
    p.push();
    p.strokeWeight(2);
    for (const e of edgeMesh) {
      if (t >= e.radius) {
        p.stroke(...colors.edgeActive);
        p.line(...e.p1, ...e.p2);
      }
    }
    p.pop();

    drawTriangleMesh(p, triangleMesh, t, colors.triangle);
    drawTriangleMesh(p, tetraMesh, t, colors.tetra);

    p.pop();
  }

  // -----------------------------
  // BUTTONS
  // -----------------------------
  document.getElementById('prevBtn_lockwasher').addEventListener('click', () => {
    tIndex = Math.max(0, tIndex - 1);
    drawSnapshot();
  });

  document.getElementById('nextBtn_lockwasher').addEventListener('click', () => {
    tIndex = Math.min(snapshots - 1, tIndex + 1);
    drawSnapshot();
  });

  // -----------------------------
  // HELPERS
  // -----------------------------
  function buildTriangleMesh(tris, pts) {
    return tris.map(tri => ({
      v1: pts[tri.i], v2: pts[tri.j], v3: pts[tri.k], radius: tri.radius
    }));
  }

  function buildTetraMesh(tetras, pts) {
    const mesh = [];
    for (const t of tetras) {
      const v = [pts[t.i], pts[t.j], pts[t.k], pts[t.l]];
      mesh.push({v1:v[0],v2:v[1],v3:v[2],radius:t.radius});
      mesh.push({v1:v[0],v2:v[1],v3:v[3],radius:t.radius});
      mesh.push({v1:v[0],v2:v[2],v3:v[3],radius:t.radius});
      mesh.push({v1:v[1],v2:v[2],v3:v[3],radius:t.radius});
    }
    return mesh;
  }

  function buildEdgeMesh(edges, pts) {
    return edges.map(e => ({
      p1: pts[e.i], p2: pts[e.j], radius: e.radius
    }));
  }

  function drawTriangleMesh(p, mesh, t, colorArr) {
    p.push();
    p.fill(...colorArr);
    p.noStroke();
    p.beginShape(p.TRIANGLES);
    for (const tri of mesh) if (t >= tri.radius) {
      p.vertex(...tri.v1); p.vertex(...tri.v2); p.vertex(...tri.v3);
    }
    p.endShape();
    p.pop();
  }

});
