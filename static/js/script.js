// Basic JavaScript to enhance interactivity

document.querySelectorAll('nav ul li a').forEach(link => {
    link.addEventListener('click', function(event) {
        event.preventDefault();
        const href = this.getAttribute('href');
        document.body.classList.add('fade-out');
        setTimeout(() => {
            window.location.href = href;
        }, 500);
    });
});

// Smooth fade-out effect
document.body.classList.add('fade-in');
/* home*/
document.addEventListener('DOMContentLoaded', function() {
    const images = document.querySelectorAll('.gallery-image');
    
    images.forEach(image => {
        image.addEventListener('click', function() {
            // Create a full-screen overlay
            const overlay = document.createElement('div');
            overlay.className = 'overlay';
            document.body.appendChild(overlay);

            // Create a larger image
            const largeImage = document.createElement('img');
            largeImage.src = this.src;
            largeImage.className = 'large-image';
            overlay.appendChild(largeImage);

            // Close overlay on click
            overlay.addEventListener('click', function() {
                document.body.removeChild(overlay);
            });
        });
    });
});

