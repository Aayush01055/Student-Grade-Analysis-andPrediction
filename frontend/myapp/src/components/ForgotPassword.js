import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { 
  Form, 
  Button, 
  Container, 
  Row, 
  Col, 
  Card, 
  Alert,
  FloatingLabel,
  Spinner
} from 'react-bootstrap';
import { Envelope, ArrowLeft } from 'react-bootstrap-icons';
import './Login.css';

const ForgotPassword = () => {
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const response = await axios.post('http://localhost:5000/api/forgot-password', {
        email
      });
      
      if (response.data.success) {
        setSuccess('Password reset link sent to your email!');
      } else {
        setError(response.data.message || 'Failed to send reset link');
      }
    } catch (err) {
      setError(err.response?.data?.message || 'An error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-background">
      <Container className="d-flex align-items-center justify-content-center" style={{ minHeight: "100vh" }}>
        <Row className="justify-content-md-center w-100">
          <Col md={6} lg={5}>
            <Card className="shadow-lg">
              <Card.Body>
                <Button 
                  variant="link" 
                  onClick={() => navigate(-1)}
                  className="mb-3 p-0 text-decoration-none"
                >
                  <ArrowLeft className="me-2" /> Back to Login
                </Button>

                <div className="text-center mb-4">
                  <Envelope size={36} className="text-primary mb-2" />
                  <h2 className="fw-bold text-primary">Reset Password</h2>
                  <p className="text-muted">
                    Enter your email to receive a password reset link
                  </p>
                </div>

                {error && <Alert variant="danger" className="text-center">{error}</Alert>}
                {success && <Alert variant="success" className="text-center">{success}</Alert>}

                <Form onSubmit={handleSubmit}>
                  <FloatingLabel controlId="email" label="Email" className="mb-3">
                    <Form.Control
                      type="email"
                      placeholder="Enter your email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                    />
                  </FloatingLabel>

                  <div className="d-grid gap-2 mt-4">
                    <Button 
                      variant="primary" 
                      type="submit" 
                      disabled={loading}
                      size="lg"
                    >
                      {loading ? (
                        <>
                          <Spinner animation="border" size="sm" className="me-2" />
                          Sending...
                        </>
                      ) : (
                        'Send Reset Link'
                      )}
                    </Button>
                  </div>
                </Form>
              </Card.Body>
            </Card>
          </Col>
        </Row>
      </Container>
    </div>
  );
};

export default ForgotPassword;