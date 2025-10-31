const API_BASE = 'http://localhost:8000';

// Disease Detection Form Handler
document.getElementById('disease-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById('image-upload');
    const resultDiv = document.getElementById('disease-result');
    
    if (!fileInput.files[0]) {
        showResult(resultDiv, '⚠️ Please select a crop image to analyze', 'error');
        return;
    }
    
    // Show loading state
    showResult(resultDiv, '🔍 Analyzing image... Please wait', 'success');
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    try {
        const response = await fetch(`${API_BASE}/predict-disease`, {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const result = await response.json();
            const confidence = (result.confidence * 100).toFixed(2);
            showResult(
                resultDiv, 
                `✅ Disease Detected: <strong>${result.disease}</strong><br>Confidence: ${confidence}%`, 
                'success'
            );
        } else {
            throw new Error('Prediction failed');
        }
    } catch (error) {
        showResult(resultDiv, '❌ Error analyzing image. Please try again or check your connection.', 'error');
        console.error('Disease prediction error:', error);
    }
});

// Crop Recommendation Form Handler
document.getElementById('recommendation-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const resultDiv = document.getElementById('recommendation-result');
    
    // Show loading state
    showResult(resultDiv, '🌱 Analyzing soil data... Please wait', 'success');
    
    const soilType = document.getElementById('soil-type').value;
    
    const data = {
        nitrogen: parseFloat(document.getElementById('nitrogen').value),
        phosphorus: parseFloat(document.getElementById('phosphorus').value),
        potassium: parseFloat(document.getElementById('potassium').value),
        temperature: parseFloat(document.getElementById('temperature').value),
        humidity: parseFloat(document.getElementById('humidity').value),
        ph: parseFloat(document.getElementById('ph').value),
        rainfall: parseFloat(document.getElementById('rainfall').value),
        soil_type: soilType || null
    };
    
    try {
        const response = await fetch(`${API_BASE}/recommend-crop`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (response.ok) {
            const result = await response.json();
            let message = `✅ Recommended Crop: <strong>${result.recommended_crop}</strong>`;
            if (result.soil_type) {
                message += `<br>Soil Type: ${result.soil_type}`;
            }
            showResult(resultDiv, message, 'success');
        } else {
            throw new Error('Recommendation failed');
        }
    } catch (error) {
        showResult(resultDiv, '❌ Error getting recommendation. Please check your input and try again.', 'error');
        console.error('Crop recommendation error:', error);
    }
});

// Weather Form Handler
document.getElementById('weather-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const location = document.getElementById('location').value;
    
    // Mock weather data (replace with actual API call)
    try {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Mock data
        const weatherData = {
            temperature: (Math.random() * 20 + 15).toFixed(1),
            humidity: (Math.random() * 40 + 40).toFixed(0),
            conditions: ['Sunny', 'Cloudy', 'Partly Cloudy', 'Rainy'][Math.floor(Math.random() * 4)],
            windSpeed: (Math.random() * 20 + 5).toFixed(1)
        };
        
        document.getElementById('temp-value').textContent = `${weatherData.temperature}°C`;
        document.getElementById('humidity-value').textContent = `${weatherData.humidity}%`;
        document.getElementById('conditions-value').textContent = weatherData.conditions;
        document.getElementById('wind-value').textContent = `${weatherData.windSpeed} km/h`;
        
    } catch (error) {
        alert('Error fetching weather data. Please try again.');
        console.error('Weather error:', error);
    }
});

// Price Form Handler
document.getElementById('price-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const crop = document.getElementById('crop-select').value;
    const location = document.getElementById('market-location').value;
    
    // Mock price data (replace with actual API call)
    try {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Mock data
        const currentPrice = (Math.random() * 3000 + 1000).toFixed(2);
        const prevPrice = (parseFloat(currentPrice) - (Math.random() * 200 - 100)).toFixed(2);
        const change = ((currentPrice - prevPrice) / prevPrice * 100).toFixed(2);
        const trend = change > 0 ? '📈 Rising' : '📉 Falling';
        
        document.getElementById('current-price').textContent = `₹${currentPrice}/quintal`;
        document.getElementById('prev-price').textContent = `₹${prevPrice}/quintal`;
        document.getElementById('price-change').textContent = `${change > 0 ? '+' : ''}${change}%`;
        document.getElementById('price-change').className = `price-value ${change > 0 ? 'positive' : 'negative'}`;
        document.getElementById('price-trend').textContent = trend;
        
    } catch (error) {
        alert('Error fetching price data. Please try again.');
        console.error('Price error:', error);
    }
});

// Utility function to display results
function showResult(div, message, type) {
    div.innerHTML = message;
    div.className = `result ${type}`;
    div.style.display = 'block';
}

// Update file upload label with filename
document.getElementById('image-upload').addEventListener('change', function(e) {
    const fileName = e.target.files[0]?.name;
    if (fileName) {
        const uploadText = document.querySelector('.upload-text');
        uploadText.textContent = fileName;
    }
});

// Crop Recommendation Form Handler
document.getElementById('recommendation-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const resultDiv = document.getElementById('recommendation-result');
    
    // Show loading state
    showResult(resultDiv, '🌱 Analyzing soil data... Please wait', 'success');
    
    const soilType = document.getElementById('soil-type').value;
    
    const data = {
        nitrogen: parseFloat(document.getElementById('nitrogen').value),
        phosphorus: parseFloat(document.getElementById('phosphorus').value),
        potassium: parseFloat(document.getElementById('potassium').value),
        temperature: parseFloat(document.getElementById('temperature').value),
        humidity: parseFloat(document.getElementById('humidity').value),
        ph: parseFloat(document.getElementById('ph').value),
        rainfall: parseFloat(document.getElementById('rainfall').value),
        soil_type: soilType || null
    };
    
    try {
        const response = await fetch(`${API_BASE}/recommend-crop`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (response.ok) {
            const result = await response.json();
            let message = `✅ Recommended Crop: <strong>${result.recommended_crop}</strong>`;
            if (result.soil_type) {
                message += `<br>Soil Type: ${result.soil_type}`;
            }
            showResult(resultDiv, message, 'success');
        } else {
            throw new Error('Recommendation failed');
        }
    } catch (error) {
        showResult(resultDiv, '❌ Error getting recommendation. Please check your input and try again.', 'error');
        console.error('Crop recommendation error:', error);
    }
});

// Utility function to display results
function showResult(div, message, type) {
    div.innerHTML = message;
    div.className = `result ${type}`;
    div.style.display = 'block';
}

// Update file upload label with filename
document.getElementById('image-upload').addEventListener('change', function(e) {
    const fileName = e.target.files[0]?.name;
    if (fileName) {
        const label = document.querySelector('.file-upload-label');
        label.textContent = `📸 ${fileName}`;
    }
});