<div class="infobox">
    <h1>MasInt</h1>
    <h2>A new computer designed to run emerging supermassive superintelligences</h2>
    <p>MasInt is founded to build supercomputers that accelerate emerging superintelligent AIs. MasInt founders are former NVIDIA employees who worked on CUDA and Cutlass. We believe that it is possible for AIs to learn faster than scaling laws by taking advantage of emerging memory technologies. MasInt phase 1 will build a prototype of a computer combining GPUs with high capacity and high bandwidth LPDDR and 3D-NAND memories to show that it is possible to significantly increase accuracy for complex engineering applications including operating system driver development, relational data analytics, and machine learning systems.</p>
</div>

<canvas id="canvas"></canvas>

<div class="controls">
    <button id="resetBtn">Reset Simulation</button>
    <button id="toggleGravityBtn">Toggle Gravity Strength</button>
</div>

<script>
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const resetBtn = document.getElementById('resetBtn');
    const toggleGravityBtn = document.getElementById('toggleGravityBtn');

    // Set canvas dimensions
    canvas.width = window.innerWidth * 0.5;
    canvas.height = window.innerHeight * 0.7;

    // Simulation parameters
    let blackHoleRadius = 30;
    let planetRadius = 10;
    let gravitationalConstant = 2.0;
    let timeSpeed = 1;
    let gravitySetting = "normal"; // normal, strong, extreme

    // Planet properties
    let planet = {
        x: canvas.width * 0.55,
        y: canvas.height * 0.9,
        vx: -0.2,
        vy: -1.0,
        trail: [],
        maxTrailLength: 150,
        captured: false,
        distortionLevel: 0
    };

    // Black hole position (center of canvas)
    const blackHoleX = canvas.width / 2;
    const blackHoleY = canvas.height / 2;

    // Accretion disk properties
    const accretionDiskInnerRadius = blackHoleRadius * 1.5;
    const accretionDiskOuterRadius = blackHoleRadius * 6;
    const accretionDiskParticles = [];
    const numParticles = 500;

    // Initialize accretion disk particles
    for (let i = 0; i < numParticles; i++) {
        const angle = Math.random() * Math.PI * 2;
        const distance = accretionDiskInnerRadius + Math.random() * (accretionDiskOuterRadius - accretionDiskInnerRadius);

        accretionDiskParticles.push({
            x: blackHoleX + Math.cos(angle) * distance,
            y: blackHoleY + Math.sin(angle) * distance,
            angle: angle,
            distance: distance,
            speed: 0.01 + (0.05 / distance),
            hue: 30 + Math.random() * 30 // Yellowish to orange color
        });
    }

    // Animation loop
    function animate() {
        // Clear canvas
        ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw stars
        drawStars();

        // Update accretion disk
        updateAccretionDisk();

        // Draw black hole
        drawBlackHole();

        // Update and draw planet
        if (!planet.captured) {
            updatePlanet();
        }
        drawPlanet();

        // Request next frame
        requestAnimationFrame(animate);
    }

    function drawStars() {
        // Draw a few stars randomly
        if (Math.random() > 0.9) {
            const x = Math.random() * canvas.width;
            const y = Math.random() * canvas.height;
            const size = Math.random() * 1.5;

            ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
            ctx.beginPath();
            ctx.arc(x, y, size, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    function updateAccretionDisk() {
        // Draw accretion disk particles
        accretionDiskParticles.forEach(particle => {
            // Update particle position (orbital motion)
            particle.angle += particle.speed;
            particle.x = blackHoleX + Math.cos(particle.angle) * particle.distance;
            particle.y = blackHoleY + Math.sin(particle.angle) * particle.distance;

            // Draw particle
            const brightness = 0.7 + Math.random() * 0.3;
            ctx.fillStyle = `hsla(${particle.hue}, 100%, ${50 + 30 * brightness}%, ${brightness})`;

            ctx.beginPath();
            ctx.arc(particle.x, particle.y, 1 + Math.random(), 0, Math.PI * 2);
            ctx.fill();
        });
    }

    function drawBlackHole() {
        // Draw event horizon (black circle)
        ctx.fillStyle = 'black';
        ctx.beginPath();
        ctx.arc(blackHoleX, blackHoleY, blackHoleRadius, 0, Math.PI * 2);
        ctx.fill();

        // Draw gravitational lensing effect
        const gradient = ctx.createRadialGradient(
            blackHoleX, blackHoleY, blackHoleRadius,
            blackHoleX, blackHoleY, blackHoleRadius * 3
        );
        gradient.addColorStop(0, 'rgba(0, 0, 0, 0.8)');
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(blackHoleX, blackHoleY, blackHoleRadius * 3, 0, Math.PI * 2);
        ctx.fill();
    }

    function updatePlanet() {
        // Calculate distance to black hole
        const dx = blackHoleX - planet.x;
        const dy = blackHoleY - planet.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        // Gravitational force (F = G * m1 * m2 / r^2)
        // We're simplifying by assuming masses are 1
        let force = gravitationalConstant / (distance * distance);

        // Apply gravitational settings
        if (gravitySetting === "strong") {
            force *= 1.5;
        } else if (gravitySetting === "extreme") {
            force *= 3;
        }

        // Calculate acceleration components
        const ax = dx / distance * force;
        const ay = dy / distance * force;

        // Update velocity
        planet.vx += ax;
        planet.vy += ay;

        // Time dilation effect - slow down near the black hole
        const timeDilation = 1 - Math.min(0.9, blackHoleRadius / distance);

        // Update position with time dilation
        planet.x += planet.vx * timeDilation * timeSpeed;
        planet.y += planet.vy * timeDilation * timeSpeed;

        // Add current position to trail
        planet.trail.push({x: planet.x, y: planet.y});

        // Limit trail length
        if (planet.trail.length > planet.maxTrailLength) {
            planet.trail.shift();
        }

        // Calculate visual distortion based on proximity
        planet.distortionLevel = Math.min(1, (blackHoleRadius * 3) / distance);

        // Check if planet has been captured (crossed event horizon)
        if (distance < blackHoleRadius) {
            planet.captured = true;
        }
    }

    function drawPlanet() {

        // Skip drawing the planet if it's been captured
        if (planet.captured) return;

        // Draw trail
        if (planet.trail.length > 1) {
            ctx.beginPath();
            ctx.moveTo(planet.trail[0].x, planet.trail[0].y);

            for (let i = 1; i < planet.trail.length; i++) {
                ctx.lineTo(planet.trail[i].x, planet.trail[i].y);
            }

            ctx.strokeStyle = 'rgba(100, 200, 255, 0.5)';
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        // Apply visual distortion to planet
        const stretchFactor = 1 + planet.distortionLevel * 2;

        // Calculate direction to black hole for stretching effect
        const dx = blackHoleX - planet.x;
        const dy = blackHoleY - planet.y;
        const angle = Math.atan2(dy, dx);

        // Draw distorted planet
        ctx.save();
        ctx.translate(planet.x, planet.y);
        ctx.rotate(angle);
        ctx.scale(stretchFactor, 1 / stretchFactor);

        // Planet gradient
        const planetGradient = ctx.createRadialGradient(0, 0, 0, 0, 0, planetRadius);
        planetGradient.addColorStop(0, '#80ff80');
        planetGradient.addColorStop(0.5, '#40cc40');
        planetGradient.addColorStop(1, '#208020');

        ctx.fillStyle = planetGradient;
        ctx.beginPath();
        ctx.arc(0, 0, planetRadius, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();
    }

    // Reset simulation
    function resetSimulation() {
        planet = {
            x: canvas.width * 0.55,
            y: canvas.height * 0.9,
            vx: -0.2,
            vy: -1.0,
            trail: [],
            maxTrailLength: 150,
            captured: false,
            distortionLevel: 0
        };
    }

    // Toggle gravity strength
    function toggleGravity() {
        if (gravitySetting === "normal") {
            gravitySetting = "strong";
            toggleGravityBtn.textContent = "Gravity: Strong";
        } else if (gravitySetting === "strong") {
            gravitySetting = "extreme";
            toggleGravityBtn.textContent = "Gravity: Extreme";
        } else {
            gravitySetting = "normal";
            toggleGravityBtn.textContent = "Gravity: Normal";
        }
    }

    // Event listeners
    resetBtn.addEventListener('click', resetSimulation);
    toggleGravityBtn.addEventListener('click', toggleGravity);

    // Handle window resize
    window.addEventListener('resize', () => {
        canvas.width = window.innerWidth * 0.5;
        canvas.height = window.innerHeight * 0.7;
        resetSimulation();
    });

    // Start animation
    animate();
</script>
