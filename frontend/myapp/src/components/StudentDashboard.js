import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { 
  Container, 
  Row, 
  Col, 
  Form, 
  Button, 
  Card, 
  Alert, 
  Spinner, 
  Badge,
  Modal
} from 'react-bootstrap';
import { 
  BarChart, 
  PersonCircle, 
  Envelope, 
  InfoCircle,
  ArrowUpCircle,
  CheckCircle,
  XCircle,
  Fullscreen,
  FullscreenExit,
  BoxArrowRight
} from 'react-bootstrap-icons';
import StudentGraph from './StudentGraph';
import './StudentDashboard.css';

const StudentDashboard = () => {
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [discrepancy, setDiscrepancy] = useState('');
  const [toEmail, setToEmail] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [showConfirmation, setShowConfirmation] = useState(false);
  const [fullscreenGraph, setFullscreenGraph] = useState(false);

  useEffect(() => {
    const checkAuth = () => {
      const token = localStorage.getItem('user');
      if (!token) {
        navigate('/');
      }
    };

    const fetchData = async () => {
      try {
        checkAuth();
        const storedUser = JSON.parse(localStorage.getItem('user'));
        
        if (storedUser && storedUser.prn) {
          setUser(storedUser);
          // Fetch any additional data if needed
        } else {
          setError('No user data found. Please log in again.');
        }
      } catch (err) {
        console.error('Error fetching data:', err);
        setError('Failed to load dashboard data');
        if (err.response?.status === 401) {
          localStorage.removeItem('token');
          localStorage.removeItem('user');
          navigate('/');
        }
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [navigate]);

  const handleLogout = () => {
    // Clear user data
    localStorage.removeItem('user');
    
    // Redirect to login page
    navigate('/');
  };

  const handleSubmitDiscrepancy = async () => {
    try {
      setLoading(true);
      const response = await axios.post('http://localhost:5000/api/submit-discrepancy', {
        prn: user.prn,
        discrepancy,
        toEmail,
      }, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (response.data.success) {
        setSuccess('Discrepancy submitted successfully!');
        setDiscrepancy('');
        setToEmail('');
        setShowConfirmation(false);
      } else {
        setError('Failed to submit discrepancy');
      }
    } catch (error) {
      console.error('Error submitting discrepancy:', error);
      setError('An error occurred while submitting discrepancy');
      if (error.response?.status === 401) {
        handleLogout();
      }
    } finally {
      setLoading(false);
    }
  };

  const renderLoading = () => (
    <div className="d-flex justify-content-center align-items-center" style={{ height: '60vh' }}>
      <Spinner animation="border" variant="primary" />
    </div>
  );

  const renderError = () => (
    <Alert variant="danger" className="text-center my-5">
      <XCircle size={24} className="me-2" />
      {error}
    </Alert>
  );

  const renderDiscrepancyForm = () => (
    <Card className="shadow-sm">
      <Card.Header className="bg-white border-bottom">
        <h5 className="mb-0">
          <Envelope className="me-2" />
          Report Discrepancy
        </h5>
      </Card.Header>
      <Card.Body>
        {success && (
          <Alert variant="success" onClose={() => setSuccess(null)} dismissible>
            <CheckCircle className="me-2" />
            {success}
          </Alert>
        )}
        {error && (
          <Alert variant="danger" onClose={() => setError(null)} dismissible>
            <XCircle className="me-2" />
            {error}
          </Alert>
        )}
        
        <Form onSubmit={(e) => { e.preventDefault(); setShowConfirmation(true); }}>
          <Form.Group controlId="toEmail" className="mb-3">
            <Form.Label className="text-muted small">Recipient Email</Form.Label>
            <Form.Control
              type="email"
              placeholder="professor@university.edu"
              value={toEmail}
              onChange={(e) => setToEmail(e.target.value)}
              required
              className="form-control-lg"
            />
          </Form.Group>
          <Form.Group controlId="discrepancy" className="mb-4">
            <Form.Label className="text-muted small">Discrepancy Details</Form.Label>
            <Form.Control
              as="textarea"
              rows={5}
              value={discrepancy}
              onChange={(e) => setDiscrepancy(e.target.value)}
              required
              className="form-control-lg"
              placeholder="Describe the issue you've identified with your marks..."
            />
          </Form.Group>
          <div className="d-flex justify-content-end">
            <Button 
              variant="primary" 
              size="lg"
              type="submit"
              disabled={loading}
            >
              {loading ? (
                <>
                  <Spinner as="span" animation="border" size="sm" className="me-2" />
                  Submitting...
                </>
              ) : (
                <>
                  <ArrowUpCircle className="me-2" />
                  Submit Report
                </>
              )}
            </Button>
          </div>
        </Form>
      </Card.Body>
    </Card>
  );

  const renderGraphSection = () => (
    <div className={`graph-container ${fullscreenGraph ? 'fullscreen' : ''}`}>
      <div className="d-flex justify-content-between align-items-center mb-3">
        <h4 className="mb-0">
          <BarChart className="me-2" />
          Performance Analysis
        </h4>
        <Button 
          variant="outline-secondary" 
          size="sm"
          onClick={() => setFullscreenGraph(!fullscreenGraph)}
          className="d-flex align-items-center"
        >
          {fullscreenGraph ? (
            <>
              <FullscreenExit className="me-1" />
              Exit Fullscreen
            </>
          ) : (
            <>
              <Fullscreen className="me-1" />
              Fullscreen
            </>
          )}
        </Button>
      </div>
      
      <Card className="shadow-sm graph-card">
        <Card.Body className="p-0">
          <div className="graph-wrapper" style={{ height: fullscreenGraph ? '75vh' : '500px' }}>
            <StudentGraph prn={user.prn} />
          </div>
        </Card.Body>
      </Card>
    </div>
  );

  if (loading) return renderLoading();
  if (error) return renderError();
  if (!user || !user.prn) {
    return (
      <Container className="mt-5 text-center">
        <InfoCircle size={48} className="text-muted mb-3" />
        <h2>Student Dashboard</h2>
        <p className="lead text-muted">Please log in to view your dashboard</p>
      </Container>
    );
  }

  return (
    <Container fluid className="student-dashboard px-lg-4 py-4">
      <Row className="mb-4">
        <Col>
          <div className="d-flex justify-content-between align-items-center">
            <div>
              <h1 className="fw-bold mb-0">
                <PersonCircle className="me-2" />
                Student Dashboard
              </h1>
              <p className="text-muted mb-0">
                Welcome back, <span className="text-primary">{user.name}</span>
              </p>
            </div>
            <div className="d-flex align-items-center gap-3">
              <Badge bg="light" text="dark" className="fs-6 border">
                PRN: {user.prn}
              </Badge>
              <Button 
                variant="outline-danger" 
                onClick={handleLogout}
                className="d-flex align-items-center"
              >
                <BoxArrowRight className="me-2" />
                Logout
              </Button>
            </div>
          </div>
        </Col>
      </Row>
      
      <Row className="g-4">
        <Col lg={8}>
          {renderGraphSection()}
        </Col>
        
        <Col lg={4}>
          {renderDiscrepancyForm()}
        </Col>
      </Row>
      
      {/* Confirmation Modal */}
      <Modal show={showConfirmation} onHide={() => setShowConfirmation(false)} centered>
        <Modal.Header closeButton>
          <Modal.Title>Confirm Discrepancy Report</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <p>Are you sure you want to submit this discrepancy report to <strong>{toEmail}</strong>?</p>
          <div className="alert alert-warning mb-0">
            <InfoCircle className="me-2" />
            Please verify all information before submission. False reports may be subject to review.
          </div>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="outline-secondary" onClick={() => setShowConfirmation(false)}>
            Cancel
          </Button>
          <Button variant="primary" onClick={handleSubmitDiscrepancy} disabled={loading}>
            {loading ? 'Submitting...' : 'Confirm Submission'}
          </Button>
        </Modal.Footer>
      </Modal>
    </Container>
  );
};

export default StudentDashboard;