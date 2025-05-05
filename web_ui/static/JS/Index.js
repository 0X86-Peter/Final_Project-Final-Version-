document.addEventListener('DOMContentLoaded', (event) => {
    const diagnoseButton = document.querySelector('.diagnosis-button'); // Make sure this class matches your button
    diagnoseButton.addEventListener('click', function() {
        window.location.href = '/recognition'; // The URL path you want to navigate to
    });
});