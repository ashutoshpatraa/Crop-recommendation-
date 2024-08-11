document.getElementById('cropForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const form = event.target;
    const formData = new FormData(form);
    const data = {};
    formData.forEach((value, key) => {
        data[key] = value;
    });

    fetch('/submit', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        const resultDiv = document.querySelector('.result');
        resultDiv.innerHTML = `<p>Recommended Crop: ${result.predicted_crop}</p>`;
    })
    .catch(error => {
        console.error('Error:', error);
    });
});