// Custom JavaScript for Student Dashboard

// Initialize tooltips
document.addEventListener('DOMContentLoaded', function() {
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});

// Chart configuration for future use
const chartColors = {
    safe: '#28a745',
    warning: '#ffc107', 
    danger: '#dc3545',
    primary: '#007bff',
    secondary: '#6c757d'
};

// Utility functions
function formatDate(dateString) {
    const options = { year: 'numeric', month: 'short', day: 'numeric' };
    return new Date(dateString).toLocaleDateString(undefined, options);
}

function getRiskColor(riskLevel) {
    switch(riskLevel) {
        case 'safe': return chartColors.safe;
        case 'warning': return chartColors.warning;
        case 'high_risk': return chartColors.danger;
        default: return chartColors.secondary;
    }
}

// Future chatbot functionality placeholder
function initChatbot() {
    console.log('Chatbot initialization placeholder');
}