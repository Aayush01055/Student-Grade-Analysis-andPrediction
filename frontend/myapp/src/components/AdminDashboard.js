import React, { useState } from 'react';
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
  Badge
} from 'react-bootstrap';
import { 
  PersonPlus, 
  PersonBadge,
  JournalBookmark,
  BoxArrowRight
} from 'react-bootstrap-icons';
import './AdminDashboard.css';

const AdminDashboard = () => {
  const navigate = useNavigate();
  const [userType, setUserType] = useState('student');
  const [formData, setFormData] = useState({
    prn: '',
    name: '',
    email: '',
    panel: '',
    rollNumber: '',
    password: '',
    department: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [showPassword, setShowPassword] = useState(false);

  const handleInputChange = (e) => {
    const { id, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [id]: value
    }));
  };

  const handleAddUser = async (e) => {
    e.preventDefault();
    try {
      setLoading(true);
      setError(null);
      
      const userData = { ...formData, userType };
      const response = await axios.post('http://localhost:5000/api/add-user', userData, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (response.data.success) {
        setSuccess(`${userType === 'student' ? 'Student' : 'Teacher'} added successfully`);
        setFormData({
          prn: '',
          name: '',
          email: '',
          panel: '',
          rollNumber: '',
          password: '',
          department: ''
        });
      } else {
        setError(response.data.message || 'Failed to add user');
      }
    } catch (error) {
      console.error('Error adding user:', error);
      setError(error.response?.data?.message || 'An error occurred');
      if (error.response?.status === 401) {
        handleLogout();
      }
    } finally {
      setLoading(false);
    }
  };

  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };

  const handleLogout = () => {
    // Clear user data
    localStorage.removeItem('user');
    
    // Redirect to login page
    navigate('/');
  };

  return (
    <Container fluid className="admin-dashboard px-lg-4 py-4">
      <Row className="mb-4">
        <Col>
          <div className="d-flex justify-content-between align-items-center">
            <div>
              <h1 className="fw-bold mb-0">Admin Dashboard</h1>
              <p className="text-muted">Add new users to the system</p>
            </div>
            <div className="d-flex align-items-center gap-3">
              <Badge bg="light" text="dark" className="fs-6 border">
                Admin
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

      <Row className="justify-content-center">
        <Col lg={8}>
          <Card className="shadow-sm">
            <Card.Header className="bg-primary text-white d-flex align-items-center">
              <PersonPlus className="me-2" size={20} />
              <h5 className="mb-0">Add {userType === 'student' ? 'Student' : 'Teacher'}</h5>
            </Card.Header>
            <Card.Body>
              {success && <Alert variant="success" onClose={() => setSuccess(null)} dismissible>{success}</Alert>}
              {error && <Alert variant="danger" onClose={() => setError(null)} dismissible>{error}</Alert>}
              
              <Form onSubmit={handleAddUser}>
                <Form.Group controlId="userType" className="mb-3">
                  <Form.Label className="text-muted small fw-bold">USER TYPE</Form.Label>
                  <div className="d-flex gap-2">
                    <Button
                      variant={userType === 'student' ? 'primary' : 'outline-primary'}
                      onClick={() => setUserType('student')}
                      className="flex-grow-1"
                    >
                      <PersonBadge className="me-2" />
                      Student
                    </Button>
                    <Button
                      variant={userType === 'teacher' ? 'primary' : 'outline-primary'}
                      onClick={() => setUserType('teacher')}
                      className="flex-grow-1"
                    >
                      <JournalBookmark className="me-2" />
                      Teacher
                    </Button>
                  </div>
                </Form.Group>

                <Row>
                  <Col md={6}>
                    <Form.Group controlId="prn" className="mb-3">
                      <Form.Label className="text-muted small fw-bold">PRN/ID</Form.Label>
                      <Form.Control
                        type="text"
                        value={formData.prn}
                        onChange={handleInputChange}
                        required
                        placeholder="Enter PRN or Staff ID"
                        className="form-control-lg"
                      />
                    </Form.Group>
                  </Col>
                  <Col md={6}>
                    <Form.Group controlId="name" className="mb-3">
                      <Form.Label className="text-muted small fw-bold">FULL NAME</Form.Label>
                      <Form.Control
                        type="text"
                        value={formData.name}
                        onChange={handleInputChange}
                        required
                        placeholder="Enter full name"
                        className="form-control-lg"
                      />
                    </Form.Group>
                  </Col>
                </Row>

                <Form.Group controlId="email" className="mb-3">
                  <Form.Label className="text-muted small fw-bold">EMAIL</Form.Label>
                  <Form.Control
                    type="email"
                    value={formData.email}
                    onChange={handleInputChange}
                    required
                    placeholder="Enter email address"
                    className="form-control-lg"
                  />
                </Form.Group>

                {userType === 'student' ? (
                  <Row>
                    <Col md={6}>
                      <Form.Group controlId="panel" className="mb-3">
                        <Form.Label className="text-muted small fw-bold">PANEL</Form.Label>
                        <Form.Control
                          type="text"
                          value={formData.panel}
                          onChange={handleInputChange}
                          required
                          placeholder="Enter panel"
                          className="form-control-lg"
                        />
                      </Form.Group>
                    </Col>
                    <Col md={6}>
                      <Form.Group controlId="rollNumber" className="mb-3">
                        <Form.Label className="text-muted small fw-bold">ROLL NUMBER</Form.Label>
                        <Form.Control
                          type="text"
                          value={formData.rollNumber}
                          onChange={handleInputChange}
                          required
                          placeholder="Enter roll number"
                          className="form-control-lg"
                        />
                      </Form.Group>
                    </Col>
                  </Row>
                ) : (
                  <Form.Group controlId="department" className="mb-3">
                    <Form.Label className="text-muted small fw-bold">DEPARTMENT</Form.Label>
                    <Form.Control
                      type="text"
                      value={formData.department}
                      onChange={handleInputChange}
                      required
                      placeholder="Enter department"
                      className="form-control-lg"
                    />
                  </Form.Group>
                )}

                <Form.Group controlId="password" className="mb-4">
                  <Form.Label className="text-muted small fw-bold">PASSWORD</Form.Label>
                  <div className="input-group">
                    <Form.Control
                      type={showPassword ? "text" : "password"}
                      value={formData.password}
                      onChange={handleInputChange}
                      required
                      placeholder="Create password"
                      className="form-control-lg"
                    />
                    <Button 
                      variant="outline-secondary" 
                      onClick={togglePasswordVisibility}
                      className="d-flex align-items-center"
                    >
                      {showPassword ? 'Hide' : 'Show'}
                    </Button>
                  </div>
                  <small className="text-muted">Minimum 8 characters</small>
                </Form.Group>

                <div className="d-grid">
                  <Button 
                    variant="primary" 
                    size="lg" 
                    type="submit"
                    disabled={loading}
                    className="d-flex align-items-center justify-content-center"
                  >
                    {loading ? (
                      <>
                        <Spinner as="span" animation="border" size="sm" className="me-2" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <PersonPlus className="me-2" />
                        Add {userType === 'student' ? 'Student' : 'Teacher'}
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
  );
};

export default AdminDashboard;