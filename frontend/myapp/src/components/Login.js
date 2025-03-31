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
  Modal,
  Spinner,
  ButtonGroup  
} from 'react-bootstrap';
import { Lock, PersonFill, PersonCheckFill, PersonBadge, PersonLinesFill, Envelope } from 'react-bootstrap-icons';
import './Login.css';

const Login = () => {
  const [userType, setUserType] = useState('student');
  const [prn, setPrn] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showForgotPassword, setShowForgotPassword] = useState(false);
  const [forgotEmail, setForgotEmail] = useState('');
  const [forgotError, setForgotError] = useState('');
  const [forgotSuccess, setForgotSuccess] = useState('');
  const [forgotLoading, setForgotLoading] = useState(false);
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      const response = await axios.post('http://localhost:5000/api/login', {
        prn,
        password,
        userType,
      });
      if (response.data.success) {
        localStorage.setItem('user', JSON.stringify(response.data.user));
        navigate(`/${userType}`);
      } else {
        setError('Invalid credentials');
      }
    } catch (error) {
      console.error('Login error:', error);
      setError('An error occurred');
    } finally {
      setLoading(false);
    }
  };

  // const handleForgotPassword = async (e) => {
  //   e.preventDefault();
  //   setForgotLoading(true);
  //   setForgotError('');
  //   setForgotSuccess('');
  //   try {
  //     const response = await axios.post('http://localhost:5000/api/forgot-password', {
  //       email: forgotEmail,
  //       userType
  //     });
  //     if (response.data.success) {
  //       setForgotSuccess('Password reset link sent to your email!');
  //     } else {
  //       setForgotError(response.data.message || 'Failed to send reset link');
  //     }
  //   } catch (error) {
  //     setForgotError('An error occurred. Please try again.');
  //   } finally {
  //     setForgotLoading(false);
  //   }
  // };

  const userTypes = [
    { value: 'student', label: 'Student', icon: <PersonFill className="me-2" /> },
    { value: 'teacher', label: 'Teacher', icon: <PersonCheckFill className="me-2" /> },
    { value: 'admin', label: 'Admin', icon: <PersonLinesFill className="me-2" /> }
  ];

  const prnPlaceholder = userType === 'student' ? 'PRN' : 'Username';

  return (
    <div className="login-background">
      <Container className="d-flex align-items-center justify-content-center" style={{ minHeight: "100vh" }}>
        <Row className="justify-content-md-center w-100">
          <Col md={6} lg={5}>
            <Card className="shadow-lg">
              <Card.Body>
                <div className="text-center mb-4">
                  <PersonBadge size={36} className="text-primary mb-2" />
                  <h2 className="fw-bold text-primary">Performance System</h2>
                  <p className="text-muted">Select your role and sign in</p>
                </div>

                {error && <Alert variant="danger">{error}</Alert>}

                <Form onSubmit={handleLogin}>
                  <div className="mb-4 text-center">
                    <ButtonGroup className="w-100">
                      {userTypes.map((type) => (
                        <Button
                          key={type.value}
                          variant={userType === type.value ? "primary" : "outline-primary"}
                          onClick={() => setUserType(type.value)}
                          className="d-flex align-items-center justify-content-center"
                        >
                          {type.icon}
                          {type.label}
                        </Button>
                      ))}
                    </ButtonGroup>
                  </div>

                  <FloatingLabel controlId="prn" label={prnPlaceholder} className="mb-3">
                    <Form.Control
                      type="text"
                      placeholder={`Enter ${prnPlaceholder}`}
                      value={prn}
                      onChange={(e) => setPrn(e.target.value)}
                      required
                    />
                  </FloatingLabel>

                  <FloatingLabel controlId="password" label="Password">
                    <Form.Control
                      type="password"
                      placeholder="Password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      required
                    />
                  </FloatingLabel>

                  <div className="d-flex justify-content-between mt-2">
                    <Form.Check 
                      type="checkbox" 
                      label="Remember me" 
                      className="text-muted"
                    />
                    <Button 
                      variant="link" 
                      className="p-0 text-decoration-none"
                      onClick={() => setShowForgotPassword(true)}
                    >
                      Forgot password?
                    </Button>
                  </div>

                  <div className="d-grid gap-2 mt-4">
                    <Button 
                      variant="primary" 
                      type="submit" 
                      disabled={loading}
                      size="lg"
                    >
                      {loading ? 'Signing in...' : (
                        <>
                          <Lock className="me-2" />
                          Sign In
                        </>
                      )}
                    </Button>
                  </div>
                </Form>
              </Card.Body>
            </Card>
          </Col>
        </Row>
      </Container>

      {/* Forgot Password Modal */}
      <Modal show={showForgotPassword} onHide={() => setShowForgotPassword(false)} centered>
        <Modal.Header closeButton>
          <Modal.Title>
            <Envelope className="me-2" />
            Reset Password
          </Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {forgotError && <Alert variant="danger">{forgotError}</Alert>}
          {forgotSuccess && <Alert variant="success">{forgotSuccess}</Alert>}

          <Form>
            <FloatingLabel controlId="forgotEmail" label="Email" className="mb-3">
              <Form.Control
                type="email"
                placeholder="Enter your email"
                value={forgotEmail}
                onChange={(e) => setForgotEmail(e.target.value)}
                required
              />
            </FloatingLabel>

            <div className="d-grid gap-2 mt-4">
              <Button 
                variant="primary" 
                type="submit" 
                disabled={forgotLoading}
                size="lg"
              >
                {forgotLoading ? (
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
        </Modal.Body>
      </Modal>
    </div>
  );
};

export default Login;