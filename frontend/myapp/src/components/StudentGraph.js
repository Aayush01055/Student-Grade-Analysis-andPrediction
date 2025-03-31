import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine } from 'recharts';

const StudentGraph = ({ prn }) => {
  const [data, setData] = useState([]);
  const [threshold, setThreshold] = useState(50);
  const [status, setStatus] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchGraphData = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await axios.get(`http://localhost:5000/api/student-graph/${prn}`);
        const { overallScore, predictedScore, threshold: backendThreshold, status: backendStatus } = response.data;

        let tempData = [
          { name: 'Current', score: overallScore },
          { name: 'Predicted', score: overallScore }, // Start prediction at current score
        ];
        setData(tempData);
        setThreshold(backendThreshold);
        setStatus(backendStatus);

        // Gradually transition to predicted score
        let step = (predictedScore - overallScore) / 20; // Smooth transition over 20 steps
        let count = 0;
        const interval = setInterval(() => {
          count++;
          tempData[1].score += step;
          setData([...tempData]);
          if (count >= 20) clearInterval(interval);
        }, 200);
      } catch (error) {
        console.error('Error fetching graph data:', error);
        setError('Failed to load student data.');
      } finally {
        setLoading(false);
      }
    };

    if (prn) fetchGraphData();
  }, [prn]);

  if (loading) return <div>Loading...</div>;
  if (error) return <div style={{ color: 'red' }}>{error}</div>;

  return (
    <div style={{ textAlign: 'center' }}>
      <h3>Student Performance (PRN: {prn})</h3>
      <LineChart
        width={500}
        height={300}
        data={data}
        margin={{ top: 20, right: 30, left: 20, bottom: 10 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis domain={[0, 100]} />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="score" stroke="#8884d8" activeDot={{ r: 8 }} name="Score" />
        <ReferenceLine y={threshold} label={`Threshold (${threshold})`} stroke="red" strokeDasharray="3 3" />
      </LineChart>
      <p style={{ color: status === 'Dropout Risk' ? 'red' : status === 'Average' ? 'orange' : 'green' }}>
        Status: {status}
      </p>
    </div>
  );
};

export default StudentGraph;
