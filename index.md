<div class="infobox">
    <h1>MasInt</h1>
    <h2>is a new computer designed to run emerging supermassive superintelligences</h2>
    <p>MasInt is founded to build supercomputers that accelerate emerging superintelligent AIs. MasInt founders are former NVIDIA employees who worked on CUDA and Cutlass. We believe that it is possible for AIs to learn faster than scaling laws by taking advantage of emerging memory technologies. MasInt phase 1 will build a prototype of a computer combining GPUs with high capacity and high bandwidth LPDDR and 3D-NAND memories to show that it is possible to significantly increase accuracy for complex engineering applications including operating system driver development, relational data analytics, and machine learning systems.</p>
</div>

<!-- Generate random stars -->
<div class="stars" id="starfield"></div>

<div class="black-hole-container">
    <!-- Background atmospheric glow -->
    <div class="disk-atmosphere"></div>

    <!-- Jet streams (vertical) -->
    <div class="jet-stream"></div>

    <!-- Upper warped disk (behind black hole) -->
    <div class="disk-upper"></div>

    <!-- Main accretion disk -->
    <div class="disk-main"></div>

    <!-- Inner bright accretion -->
    <div class="inner-glow"></div>

    <!-- Doppler beaming bright spot -->
    <div class="doppler-bright"></div>

    <!-- Photon sphere -->
    <div class="photon-sphere"></div>

    <!-- Event horizon (black hole) -->
    <div class="event-horizon"></div>

    <!-- Lower warped disk (in front) -->
    <div class="disk-lower"></div>
</div>

<script>
    // Generate random stars with varying brightness
    const starfield = document.getElementById('starfield');
    const numStars = 150;

    for (let i = 0; i < numStars; i++) {
        const star = document.createElement('div');
        star.className = 'star';
        star.style.left = Math.random() * 100 + '%';
        star.style.top = Math.random() * 100 + '%';
        star.style.animationDelay = Math.random() * 4 + 's';

        // Vary star sizes and brightness
        const size = Math.random() * 2;
        star.style.width = size + 'px';
        star.style.height = size + 'px';
        star.style.opacity = Math.random() * 0.8 + 0.2;

        starfield.appendChild(star);
    }
</script>
