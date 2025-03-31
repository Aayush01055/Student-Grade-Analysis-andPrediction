import React, { useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
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
import { Key, ArrowLeft } from 'react-bootstrap-icons';
import './Login.css';

const ResetPassword = () => {
  const [searchParams] = useSearchParams();
  const token = searchParams.get('token');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const response = await axios.post('http://localhost:5000/api/reset-password', {
        token,
        newPassword: password
      });
      
      if (response.data.success) {
        setSuccess('Password reset successfully!');
        setTimeout(() => navigate('/login'), 2000);
      } else {
        setError(response.data.message || 'Failed to reset password');
      }
    } catch (err) {
      setError(err.response?.data?.message || 'An error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  if (!token) {
    return (
      <div className="login-background">
        <Container className="d-flex align-items-center justify-content-center" style={{ minHeight: "100vh" }}>
          <Alert variant="danger" className="text-center">
            Invalid or missing reset token
          </Alert>
        </Container>
      </div>
    );
  }

  return (
    <div className="login-background">
      <Container className="d-flex align-items-center justify-content-center" style={{ minHeight: "100vh" }}>
        <Row className="justify-content-md-center w-100">
          <Col md={6} lg={5}>
            <Card className="shadow-lg">
              <Card.Body>
                <Button 
                  variant="link" 
                  onClick={() => navigate('/login')}
                  className="mb-3 p-0 text-decoration-none"
                >
                  <ArrowLeft className="me-2" /> Back to Login
                </Button>

                <div className="text-center mb-4">
                  <Key size={36} className="text-primary mb-2" />
                  <h2 className="fw-bold text-primary">Set New Password</h2>
                  <p className="text-muted">
                    Enter your new password below
                  </p>
                </div>

                {error && <Alert variant="danger" className="text-center">{error}</Alert>}
                {success && <Alert variant="success" className="text-center">{success}</Alert>}

                <Form onSubmit={handleSubmit}>
                  <FloatingLabel controlId="password" label="New Password" className="mb-3">
                    <Form.Control
                      type="password"
                      placeholder="Enter new password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      required
                      minLength={8}
                    />
                  </FloatingLabel>

                  <FloatingLabel controlId="confirmPassword" label="Confirm Password">
                    <Form.Control
                      type="password"
                      placeholder="Confirm new password"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      required
                      minLength={8}
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
                          Resetting...
                        </>
                      ) : (
                        'Reset Password'
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

export default ResetPassword;