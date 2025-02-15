document.addEventListener('DOMContentLoaded', () => {
    const textInput = document.getElementById('text-input');
    const currentCount = document.getElementById('current-count');

    // Character count update
    textInput.addEventListener('input', () => {
        const length = textInput.value.length;
        currentCount.textContent = length;
    });
});

document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const text = document.getElementById('text-input').value.trim();
    const predictButton = document.getElementById('predict-button');
    const buttonText = predictButton.querySelector('span');
    const loader = predictButton.querySelector('.loader');
    const resultSection = document.getElementById('result');

    if (!text) {
        alert('Please share your thoughts first!');
        return;
    }

    // Disable button and show loader
    predictButton.disabled = true;
    loader.style.display = 'block';
    buttonText.textContent = 'Analyzing...';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text }),
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const data = await response.json();

        // Display simple result
        document.getElementById('mbti-type').textContent = data.mbti_type;
        document.getElementById('confidence').textContent = 
            `${(data.confidence * 100).toFixed(0)}% Confidence Score`;

        // Show result
        resultSection.style.display = 'block';
        requestAnimationFrame(() => {
            resultSection.classList.remove('hidden');
        });
    } catch (error) {
        console.error('Error:', error);
        alert('Oops! Something went wrong. Please try again.');
    } finally {
        // Reset button state
        predictButton.disabled = false;
        loader.style.display = 'none';
        buttonText.textContent = 'Analyze Personality';
    }
});
