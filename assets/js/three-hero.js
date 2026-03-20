/**
 * Three.js Particle Network — Home page hero background
 * Creates an animated particle network with violet and teal particles
 * connected by lines, with subtle mouse interaction.
 */
(function() {
  'use strict';

  // Respect reduced motion
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  function initThreeHero() {
    var container = document.getElementById('hero-canvas');
    if (!container) return;

    // Check for Three.js availability
    if (typeof THREE === 'undefined') {
      // Fallback: CSS gradient background
      container.style.background = 'radial-gradient(ellipse at 60% 40%, rgba(108,43,217,0.12) 0%, transparent 60%)';
      return;
    }

    // Check for WebGL support
    try {
      var testCanvas = document.createElement('canvas');
      var gl = testCanvas.getContext('webgl') || testCanvas.getContext('experimental-webgl');
      if (!gl) throw new Error('No WebGL');
    } catch(e) {
      container.style.background = 'radial-gradient(ellipse at 60% 40%, rgba(108,43,217,0.12) 0%, transparent 60%)';
      return;
    }

    var scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    var renderer;

    try {
      renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    } catch(e) {
      container.style.background = 'radial-gradient(ellipse at 60% 40%, rgba(108,43,217,0.12) 0%, transparent 60%)';
      return;
    }

    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    // Configuration
    var isMobile = window.innerWidth < 768;
    var particleCount = isMobile ? 80 : 150;
    var spread = 15;
    var connectionDistance = isMobile ? 3 : 2.5;

    // Create particles
    var positions = new Float32Array(particleCount * 3);
    var colorAttr = new Float32Array(particleCount * 3);
    var velocities = [];
    var violet = new THREE.Color(0x6C2BD9);
    var teal = new THREE.Color(0x00D4AA);

    for (var i = 0; i < particleCount; i++) {
      positions[i * 3] = (Math.random() - 0.5) * spread * 2;
      positions[i * 3 + 1] = (Math.random() - 0.5) * spread * 2;
      positions[i * 3 + 2] = (Math.random() - 0.5) * spread;

      velocities.push({
        x: (Math.random() - 0.5) * 0.005,
        y: (Math.random() - 0.5) * 0.005,
        z: (Math.random() - 0.5) * 0.003
      });

      // Alternate colors: violet and teal
      var c = Math.random() > 0.5 ? violet : teal;
      colorAttr[i * 3] = c.r;
      colorAttr[i * 3 + 1] = c.g;
      colorAttr[i * 3 + 2] = c.b;
    }

    var geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colorAttr, 3));

    var material = new THREE.PointsMaterial({
      size: isMobile ? 0.06 : 0.04,
      transparent: true,
      opacity: 0.7,
      sizeAttenuation: true,
      vertexColors: true
    });

    var points = new THREE.Points(geometry, material);
    scene.add(points);

    // Lines connecting nearby particles
    var lineGeometry = new THREE.BufferGeometry();
    var maxLines = particleCount * 3;
    var linePositions = new Float32Array(maxLines * 6);
    lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
    lineGeometry.setDrawRange(0, 0);

    var lineMaterial = new THREE.LineBasicMaterial({
      color: 0xA78BFA,
      transparent: true,
      opacity: 0.12
    });

    var lines = new THREE.LineSegments(lineGeometry, lineMaterial);
    scene.add(lines);

    camera.position.z = 12;

    // Mouse tracking
    var mouseX = 0, mouseY = 0;
    document.addEventListener('mousemove', function(e) {
      mouseX = (e.clientX / window.innerWidth) * 2 - 1;
      mouseY = -(e.clientY / window.innerHeight) * 2 + 1;
    });

    // Animation loop
    function animate() {
      requestAnimationFrame(animate);

      var pos = geometry.attributes.position.array;
      var lineIdx = 0;
      var linePos = lineGeometry.attributes.position.array;

      for (var i = 0; i < particleCount; i++) {
        // Mouse repulsion (gentle)
        var px = pos[i * 3];
        var py = pos[i * 3 + 1];
        var mx = mouseX * spread;
        var my = mouseY * spread;
        var dmx = px - mx;
        var dmy = py - my;
        var distMouse = Math.sqrt(dmx * dmx + dmy * dmy);
        if (distMouse < 3) {
          var force = (3 - distMouse) * 0.001;
          velocities[i].x += dmx * force;
          velocities[i].y += dmy * force;
        }

        // Dampen velocity
        velocities[i].x *= 0.999;
        velocities[i].y *= 0.999;

        pos[i * 3] += velocities[i].x;
        pos[i * 3 + 1] += velocities[i].y;
        pos[i * 3 + 2] += velocities[i].z;

        // Bounds check
        if (Math.abs(pos[i * 3]) > spread) velocities[i].x *= -1;
        if (Math.abs(pos[i * 3 + 1]) > spread) velocities[i].y *= -1;
        if (Math.abs(pos[i * 3 + 2]) > spread * 0.5) velocities[i].z *= -1;

        // Draw connections
        for (var j = i + 1; j < particleCount && lineIdx < maxLines; j++) {
          var dx = pos[i * 3] - pos[j * 3];
          var dy = pos[i * 3 + 1] - pos[j * 3 + 1];
          var dz = pos[i * 3 + 2] - pos[j * 3 + 2];
          var dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

          if (dist < connectionDistance) {
            linePos[lineIdx * 6] = pos[i * 3];
            linePos[lineIdx * 6 + 1] = pos[i * 3 + 1];
            linePos[lineIdx * 6 + 2] = pos[i * 3 + 2];
            linePos[lineIdx * 6 + 3] = pos[j * 3];
            linePos[lineIdx * 6 + 4] = pos[j * 3 + 1];
            linePos[lineIdx * 6 + 5] = pos[j * 3 + 2];
            lineIdx++;
          }
        }
      }

      geometry.attributes.position.needsUpdate = true;
      lineGeometry.attributes.position.needsUpdate = true;
      lineGeometry.setDrawRange(0, lineIdx * 2);

      // Subtle camera follow mouse
      camera.position.x += (mouseX * 1.5 - camera.position.x) * 0.02;
      camera.position.y += (mouseY * 1.5 - camera.position.y) * 0.02;
      camera.lookAt(scene.position);

      renderer.render(scene, camera);
    }

    animate();

    // Responsive resize
    window.addEventListener('resize', function() {
      if (!container.clientWidth || !container.clientHeight) return;
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
      setTimeout(initThreeHero, 100);
    });
  } else {
    setTimeout(initThreeHero, 100);
  }
})();
