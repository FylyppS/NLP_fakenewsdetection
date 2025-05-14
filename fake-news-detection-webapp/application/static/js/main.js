document.addEventListener('DOMContentLoaded', function() {
    // Input type toggle
    const inputTypeSelector = document.getElementById('input-type');
    const textInputGroup = document.getElementById('text-input-group');
    const urlInputGroup = document.getElementById('url-input-group');
    
    inputTypeSelector.addEventListener('change', function() {
        if (this.value === 'text') {
            textInputGroup.classList.remove('d-none');
            urlInputGroup.classList.add('d-none');
        } else {
            textInputGroup.classList.add('d-none');
            urlInputGroup.classList.remove('d-none');
        }
    });
    
    // Form submission
    const detectionForm = document.getElementById('detection-form');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultsCard = document.getElementById('results-card');
    const errorMessage = document.getElementById('error-message');
    const loadingIndicator = document.getElementById('loading');
    
    detectionForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Hide any previous results or errors
        resultsCard.classList.add('d-none');
        errorMessage.classList.add('d-none');
        
        // Show loading indicator
        loadingIndicator.classList.remove('d-none');
        analyzeBtn.disabled = true;
        
        // Get input data
        const inputType = inputTypeSelector.value;
        let text = '';
        let url = '';
        
        if (inputType === 'text') {
            text = document.getElementById('news-text').value.trim();
            if (!text) {
                showError('Please enter some text to analyze.');
                return;
            }
        } else {
            url = document.getElementById('news-url').value.trim();
            if (!url) {
                showError('Please enter a URL to analyze.');
                return;
            }
        }
        
        try {
            // Prepare request data
            const requestData = {
                message: {
                    text: text,
                    url: url || null
                }
            };
            
            // Send API request
            const response = await fetch('/api/v1/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'An error occurred while analyzing the text.');
            }
            
            const analysisData = await response.json();
            displayResults(analysisData);
            
        } catch (error) {
            showError(error.message || 'An unexpected error occurred.');
        } finally {
            // Hide loading indicator
            loadingIndicator.classList.add('d-none');
            analyzeBtn.disabled = false;
        }
    });
    
    function displayResults(data) {
        // Display prediction result
        const resultElement = document.getElementById('prediction-result');
        resultElement.textContent = data.prediction.predicted_label.toUpperCase();
        
        // Set appropriate color based on prediction
        if (data.prediction.predicted_label === 'agree') {
            resultElement.className = 'mb-3 text-success';
        } else if (data.prediction.predicted_label === 'disagree') {
            resultElement.className = 'mb-3 text-danger';
        } else if (data.prediction.predicted_label === 'discuss') {
            resultElement.className = 'mb-3 text-primary';
        } else {
            resultElement.className = 'mb-3 text-warning';
        }
        
        // Update confidence bar
        const confidencePercent = Math.round(data.prediction.confidence * 100);
        document.getElementById('confidence-bar').style.width = `${confidencePercent}%`;
        document.getElementById('confidence-text').textContent = `Confidence: ${confidencePercent}%`;
        
        // Update keywords
        const keywordsContainer = document.getElementById('keywords');
        keywordsContainer.innerHTML = '';
        
        if (data.keywords && data.keywords.length) {
            data.keywords.forEach(keyword => {
                const badge = document.createElement('span');
                badge.className = 'badge bg-secondary me-1 mb-1';
                badge.textContent = keyword;
                keywordsContainer.appendChild(badge);
            });
        } else {
            keywordsContainer.textContent = 'No keywords extracted';
        }
        
        // Update statistics
        document.getElementById('word-count').textContent = data.statistics.word_count;
        document.getElementById('char-count').textContent = data.statistics.char_count;
        document.getElementById('sentence-count').textContent = data.statistics.sentence_count;
        
        // Update sentiment bars
        document.getElementById('positive-bar').style.width = `${data.sentiment.pos * 100}%`;
        document.getElementById('neutral-bar').style.width = `${data.sentiment.neu * 100}%`;
        document.getElementById('negative-bar').style.width = `${data.sentiment.neg * 100}%`;
        
        // Show results card
        resultsCard.classList.remove('d-none');
        
        // Scroll to results
        resultsCard.scrollIntoView({behavior: 'smooth'});
    }
    
    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.classList.remove('d-none');
        loadingIndicator.classList.add('d-none');
        analyzeBtn.disabled = false;
    }
});