import React, { useState } from 'react';
import axios from 'axios';
import { Form, Button } from 'react-bootstrap';

const FeedbackForm = ({ prn, email }) => {
  const [feedback, setFeedback] = useState('');

  const handleSendFeedback = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:5000/api/send-feedback', {
        prn,
        email,
        feedback,
      });
      if (response.data.success) {
        alert('Feedback sent successfully');
        setFeedback('');
      } else {
        alert('Failed to send feedback');
      }
    } catch (error) {
      console.error('Error sending feedback:', error);
      alert('An error occurred');
    }
  };

  return (
    <Form onSubmit={handleSendFeedback}>
      <Form.Group controlId="feedback">
        <Form.Label>Feedback/Suggestion</Form.Label>
        <Form.Control
          as="textarea"
          rows={3}
          value={feedback}
          onChange={(e) => setFeedback(e.target.value)}
          required
        />
      </Form.Group>
      <Button variant="primary" type="submit" className="mt-3">
        Send Feedback
      </Button>
    </Form>
  );
};

export default FeedbackForm;