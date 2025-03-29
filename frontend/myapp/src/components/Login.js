import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { Form, Button, Container, Row, Col } from 'react-bootstrap';

const Login = () => {
  const [userType, setUserType] = useState('student');
  const [prn, setPrn] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:5000/api/login', {
        prn,
        password,
        userType,
      });
      if (response.data.success) {
        localStorage.setItem('user', JSON.stringify(response.data.user));
        if (userType === 'admin') navigate('/admin');
        else if (userType === 'teacher') navigate('/teacher');
        else navigate('/student');
      } else {
        alert('Invalid credentials');
      }
    } catch (error) {
      console.error('Login error:', error);
      alert('An error occurred');
    }
  };

  return (
    <Container className="mt-5">
      <Row className="justify-content-md-center">
        <Col md={4}>
          <h2>Login</h2>
          <Form onSubmit={handleLogin}>
            <Form.Group controlId="userType">
              <Form.Label>User Type</Form.Label>
              <Form.Control
                as="select"
                value={userType}
                onChange={(e) => setUserType(e.target.value)}
              >
                <option value="student">Student</option>
                <option value="teacher">Teacher</option>
                <option value="admin">Admin</option>
              </Form.Control>
            </Form.Group>
            <Form.Group controlId="prn" className="mt-3">
              <Form.Label>PRN</Form.Label>
              <Form.Control
                type="text"
                value={prn}
                onChange={(e) => setPrn(e.target.value)}
                required
              />
            </Form.Group>
            <Form.Group controlId="password" className="mt-3">
              <Form.Label>Password</Form.Label>
              <Form.Control
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </Form.Group>
            <Button variant="primary" type="submit" className="mt-4">
              Login
            </Button>
          </Form>
        </Col>
      </Row>
    </Container>
  );
};

export default Login;