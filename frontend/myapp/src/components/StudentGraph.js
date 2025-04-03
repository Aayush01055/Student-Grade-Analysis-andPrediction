import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine } from 'recharts';

const StudentGraph = ({ prn }) => {
  const [graphData, setGraphData] = useState({
    dataPoints: [],
    threshold: 50,
    status: '',
    remark: '',
    hasRemark: false,
    isLoading: true,
    error: null,
    noData: false
  });

  useEffect(() => {
    const fetchGraphData = async () => {
      try {
        const response = await axios.get(`http://localhost:5000/api/student-graph/${prn}`);
        
        const newState = {
          isLoading: false,
          error: null,
          noData: response.data.status === 'No Data',
          threshold: response.data.threshold,
          status: response.data.status,
          remark: response.data.remark,
          hasRemark: response.data.hasRemark,
          dataPoints: []
        };

        if (!newState.noData) {
          newState.dataPoints = [
            { name: 'Current', score: response.data.overallScore },
            { name: 'Predicted', score: response.data.overallScore }
          ];

          // Animate prediction transition
          let tempData = [...newState.dataPoints];
          let step = (response.data.predictedScore - response.data.overallScore) / 20;
          let count = 0;
          const interval = setInterval(() => {
            count++;
            tempData[1].score += step;
            setGraphData(prev => ({ ...prev, dataPoints: [...tempData] }));
            if (count >= 20) clearInterval(interval);
          }, 200);
        }

        setGraphData(newState);
      } catch (error) {
        setGraphData(prev => ({
          ...prev,
          isLoading: false,
          error: 'Failed to load student data',
          remark: 'Error loading remarks'
        }));
      }
    };

    if (prn) fetchGraphData();
  }, [prn]);

  // Function to split remarks by semicolon and trim whitespace
  const splitRemarks = (remark) => {
    if (!remark) return [];
    return remark.split(';').map(item => item.trim()).filter(item => item.length > 0);
  };

  if (graphData.isLoading) return <div className="loading">Loading...</div>;
  if (graphData.error) return <div className="error">{graphData.error}</div>;

  return (
    <div className="student-graph-container">
      <h3>Student Performance (PRN: {prn})</h3>
      
      {graphData.noData ? (
        <div className="no-data">
          <p>No data available for this student</p>
          <p className="no-data-remark">
            This student has no marks recorded in the system yet.
          </p>
        </div>
      ) : (
        <>
          <LineChart
            width={500}
            height={300}
            data={graphData.dataPoints}
            margin={{ top: 20, right: 30, left: 20, bottom: 10 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis domain={[0, 100]} />
            <Tooltip formatter={(value) => [`${value.toFixed(1)}%`, 'Score']} />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="score" 
              stroke="#8884d8" 
              strokeWidth={2}
              activeDot={{ r: 8 }} 
              name="Score" 
            />
            <ReferenceLine 
              y={graphData.threshold} 
              label={`Threshold (${graphData.threshold}%)`} 
              stroke="red" 
              strokeDasharray="3 3" 
            />
          </LineChart>
          <p className={`status ${graphData.status.toLowerCase().replace(' ', '-')}`}>
            Status: {graphData.status}
          </p>
        </>
      )}

      {/* Remark section - only show if there's a remark or in error state */}
      {(graphData.hasRemark || graphData.error) && (
        <div className="remark-box">
          <h4>Teacher's Remarks:</h4>
          <ul className="remark-list">
            {splitRemarks(graphData.remark).map((item, index) => (
              <li key={index} className="remark-item">{item}</li>
            ))}
          </ul>
        </div>
      )}

      <style jsx>{`
        .student-graph-container {
          text-align: center;
          padding: 20px;
          max-width: 600px;
          margin: 0 auto;
        }
        .loading, .error {
          padding: 20px;
          text-align: center;
        }
        .error {
          color: red;
        }
        .no-data {
          color: #666;
        }
        .no-data-remark {
          font-style: italic;
        }
        .status {
          font-weight: bold;
          margin: 15px 0;
        }
        .status.excellent { color: green; }
        .status.good { color: blue; }
        .status.average { color: orange; }
        .status.needs-improvement { color: red; }
        .remark-box {
          margin-top: 25px;
          padding: 15px;
          background: #f8f9fa;
          border-radius: 8px;
          text-align: left;
          border-left: 4px solid #6c757d;
        }
        .remark-list {
          padding-left: 20px;
          margin: 10px 0 5px;
        }
        .remark-item {
          color: red;
          margin-bottom: 5px;
          list-style-type: disc;
        }
      `}</style>
    </div>
  );
};

export default StudentGraph;