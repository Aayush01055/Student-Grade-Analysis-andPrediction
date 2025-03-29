import React, { useState } from 'react';
import axios from 'axios';
import { Form, Button, Container, Row, Col } from 'react-bootstrap';

const AdminDashboard = () => {
  const [userType, setUserType] = useState('student');
  const [prn, setPrn] = useState('');
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [panel, setPanel] = useState('');
  const [rollNumber, setRollNumber] = useState('');
  const [password, setPassword] = useState('');

  const handleAddUser = async (e) => {
    e.preventDefault();
    try {
      const userData = { prn, name, email, panel, rollNumber, password, userType };
      const response = await axios.post('http://localhost:5000/api/add-user', userData);
      if (response.data.success) {
        alert(`${userType} added successfully`);
        setPrn('');
        setName('');
        setEmail('');
        setPanel('');
        setRollNumber('');
        setPassword('');
      } else {
        alert('Failed to add user');
      }
    } catch (error) {
      console.error('Error adding user:', error);
      alert('An error occurred');
    }
  };

  return (
    <Container className="mt-5">
      <h2>Admin Dashboard</h2>
      <Row className="justify-content-md-center">
        <Col md={6}>
          <h4>Add {userType === 'student' ? 'Student' : 'Teacher'}</h4>
          <Form onSubmit={handleAddUser}>
            <Form.Group controlId="userType">
              <Form.Label>User Type</Form.Label>
              <Form.Control
                as="select"
                value={userType}
                onChange={(e) => setUserType(e.target.value)}
              >
                <option value="student">Student</option>
                <option value="teacher">Teacher</option>
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
            <Form.Group controlId="name" className="mt-3">
              <Form.Label>Name</Form.Label>
              <Form.Control
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
              />
            </Form.Group>
            <Form.Group controlId="email" className="mt-3">
              <Form.Label>Email</Form.Label>
              <Form.Control
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </Form.Group>
            {userType === 'student' && (
              <>
                <Form.Group controlId="panel" className="mt-3">
                  <Form.Label>Panel</Form.Label>
                  <Form.Control
                    type="text"
                    value={panel}
                    onChange={(e) => setPanel(e.target.value)}
                    required
                  />
                </Form.Group>
                <Form.Group controlId="rollNumber" className="mt-3">
                  <Form.Label>Roll Number</Form.Label>
                  <Form.Control
                    type="text"
                    value={rollNumber}
                    onChange={(e) => setRollNumber(e.target.value)}
                    required
                  />
                </Form.Group>
              </>
            )}
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
              Add {userType === 'student' ? 'Student' : 'Teacher'}
            </Button>
          </Form>
        </Col>
      </Row>
    </Container>
  );
};

export default AdminDashboard;