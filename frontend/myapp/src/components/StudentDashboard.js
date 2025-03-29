import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Container, Row, Col, Form, Button } from 'react-bootstrap';
import StudentGraph from './StudentGraph';

const StudentDashboard = () => {
  const [user, setUser] = useState(null);
  const [discrepancy, setDiscrepancy] = useState('');
  const [toEmail, setToEmail] = useState(''); // New state for the "To" email
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const storedUser = JSON.parse(localStorage.getItem('user'));
    console.log('Stored User:', storedUser); // Debug: Check if user data is retrieved
    if (storedUser && storedUser.prn) {
      setUser(storedUser);
    } else {
      setError('No user data found. Please log in again.');
    }
    setLoading(false);
  }, []);

  const handleSubmitDiscrepancy = async (e) => {
    e.preventDefault();
    if (!user || !user.prn) {
      alert('User not authenticated. Please log in.');
      return;
    }

    if (!toEmail) {
      alert('Please enter a recipient email address.');
      return;
    }

    try {
      const response = await axios.post('http://localhost:5000/api/submit-discrepancy', {
        prn: user.prn,
        discrepancy,
        toEmail, // Include the "To" email in the request
      });
      if (response.data.success) {
        alert('Discrepancy submitted successfully');
        setDiscrepancy('');
        setToEmail(''); // Clear the email field after submission
      } else {
        alert('Failed to submit discrepancy');
      }
    } catch (error) {
      console.error('Error submitting discrepancy:', error);
      alert('An error occurred while submitting discrepancy');
    }
  };

  if (loading) {
    return (
      <Container className="mt-5">
        <h2>Loading...</h2>
      </Container>
    );
  }

  if (error) {
    return (
      <Container className="mt-5">
        <h2>Error</h2>
        <p>{error}</p>
      </Container>
    );
  }

  if (!user || !user.prn) {
    return (
      <Container className="mt-5">
        <h2>Student Dashboard</h2>
        <p>Please log in to view your dashboard.</p>
      </Container>
    );
  }

  return (
    <Container className="mt-5">
      <h2>Student Dashboard</h2>
      <Row>
        <Col md={6}>
          <h4>Your Performance Graph</h4>
          <StudentGraph prn={user.prn} />
        </Col>
        <Col md={6}>
          <h4>Submit Discrepancy</h4>
          <Form onSubmit={handleSubmitDiscrepancy}>
            <Form.Group controlId="toEmail" className="mb-3">
              <Form.Label>Recipient Email Address</Form.Label>
              <Form.Control
                type="email"
                placeholder="Enter recipient email (e.g., admin@example.com)"
                value={toEmail}
                onChange={(e) => setToEmail(e.target.value)}
                required
              />
            </Form.Group>
            <Form.Group controlId="discrepancy" className="mb-3">
              <Form.Label>Describe the Discrepancy</Form.Label>
              <Form.Control
                as="textarea"
                rows={3}
                value={discrepancy}
                onChange={(e) => setDiscrepancy(e.target.value)}
                required
              />
            </Form.Group>
            <Button variant="primary" type="submit">
              Submit Discrepancy
            </Button>
          </Form>
        </Col>
      </Row>
    </Container>
  );
};

export default StudentDashboard;