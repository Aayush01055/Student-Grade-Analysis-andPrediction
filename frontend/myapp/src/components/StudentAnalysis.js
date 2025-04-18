import React from 'react';
import { useNavigate } from 'react-router-dom';

function StudentAnalysis() {
    const navigate = useNavigate();

    const handleBackClick = () => {
        navigate('/teacher'); // Replace with the actual route for the teacher dashboard
    };

    return (
        <div style={{ position: 'relative', width: '100vw', height: '100vh' }}>
            <button
                onClick={handleBackClick}
                style={{
                    position: 'absolute',
                    top: '10px',
                    left: '10px',
                    zIndex: 1000,
                    padding: '10px 20px',
                    backgroundColor: '#007BFF',
                    color: '#fff',
                    border: 'none',
                    borderRadius: '5px',
                    cursor: 'pointer',
                }}
            >
                Back
            </button>
            <iframe
                src={`${process.env.PUBLIC_URL}/analysis.html`}
                title="Embedded Page"
                width="100%"
                height="100%"
                style={{
                    border: 'none',
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                }}
            />
        </div>
    );
}

export default StudentAnalysis;