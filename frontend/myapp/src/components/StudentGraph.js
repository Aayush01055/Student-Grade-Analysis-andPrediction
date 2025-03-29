import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine } from 'recharts';

const StudentGraph = ({ prn }) => {
  const [data, setData] = useState([]);
  const [threshold, setThreshold] = useState(50); // Threshold set to 30

  useEffect(() => {
    const fetchGraphData = async () => {
      try {
        const response = await axios.get(`http://localhost:5000/api/student-graph/${prn}`);
        const { overallScore, predictedScore, threshold: backendThreshold } = response.data;
        setData([
          { name: 'Current', score: overallScore },
          { name: 'Predicted', score: predictedScore },
        ]);
        setThreshold(backendThreshold); // Use the threshold from the backend
      } catch (error) {
        console.error('Error fetching graph data:', error);
      }
    };
    fetchGraphData();
  }, [prn]);

  return (
    <div>
      <LineChart width={500} height={300} data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis domain={[0, 100]} />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="score" stroke="#8884d8" />
        <ReferenceLine y={threshold} label="Threshold (50)" stroke="red" strokeDasharray="3 3" />
      </LineChart>
      <p>
        {data[0]?.score < threshold ? 'Dropout Risk' : data[0]?.score === threshold ? 'Average' : 'Excellent'}
      </p>
    </div>
  );
};

export default StudentGraph;