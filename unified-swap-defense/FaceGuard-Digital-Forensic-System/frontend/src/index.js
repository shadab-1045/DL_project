import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import FrontendTracer from './FrontendTracer';

// Initialize OpenTelemetry
if (typeof window !== 'undefined') {
  FrontendTracer();
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<React.StrictMode><App /></React.StrictMode>);
