import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Container, Row, Col, Table, Button, Form } from 'react-bootstrap';
import StudentGraph from './StudentGraph';
import FeedbackForm from './FeedbackForm';

const TeacherDashboard = () => {
  const [students, setStudents] = useState([]);
  const [selectedStudent, setSelectedStudent] = useState(null);
  const [marks, setMarks] = useState({
    cca1: '',
    cca2: '',
    cca3: '',
    lca1: '',
    lca2: '',
    lca3: '',
    co1: '',
    co2: '',
    co3: '',
    co4: '',
  });

  useEffect(() => {
    const fetchStudents = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/students');
        setStudents(response.data.students);
      } catch (error) {
        console.error('Error fetching students:', error);
      }
    };
    fetchStudents();
  }, []);

  const handleAddMarks = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:5000/api/add-marks', {
        prn: selectedStudent.prn,
        ...marks,
      });
      if (response.data.success) {
        alert('Marks added successfully');
        setMarks({
          cca1: '',
          cca2: '',
          cca3: '',
          lca1: '',
          lca2: '',
          lca3: '',
          co1: '',
          co2: '',
          co3: '',
          co4: '',
        });
      } else {
        alert('Failed to add marks');
      }
    } catch (error) {
      console.error('Error adding marks:', error);
      alert('An error occurred');
    }
  };

  return (
    <Container className="mt-5">
      <h2>Teacher Dashboard</h2>
      <Row>
        <Col md={6}>
          <h4>Student List</h4>
          <Table striped bordered hover>
            <thead>
              <tr>
                <th>PRN</th>
                <th>Name</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {students.map((student) => (
                <tr key={student.prn}>
                  <td>{student.prn}</td>
                  <td>{student.name}</td>
                  <td>
                    <Button
                      variant="info"
                      onClick={() => setSelectedStudent(student)}
                    >
                      View Details
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </Table>
        </Col>
        {selectedStudent && (
          <Col md={6}>
            <h4>Student Details: {selectedStudent.name}</h4>
            <h5>Add Marks</h5>
            <Form onSubmit={handleAddMarks}>
              <Form.Group controlId="cca1">
                <Form.Label>CCA-1 (10 marks)</Form.Label>
                <Form.Control
                  type="number"
                  value={marks.cca1}
                  onChange={(e) => setMarks({ ...marks, cca1: e.target.value })}
                  required
                />
              </Form.Group>
              <Form.Group controlId="cca2" className="mt-3">
                <Form.Label>CCA-2 (5 marks)</Form.Label>
                <Form.Control
                  type="number"
                  value={marks.cca2}
                  onChange={(e) => setMarks({ ...marks, cca2: e.target.value })}
                  required
                />
              </Form.Group>
              <Form.Group controlId="cca3" className="mt-3">
                <Form.Label>CCA-3 (Mid term - 15 marks)</Form.Label>
                <Form.Control
                  type="number"
                  value={marks.cca3}
                  onChange={(e) => setMarks({ ...marks, cca3: e.target.value })}
                  required
                />
              </Form.Group>
              <Form.Group controlId="lca1" className="mt-3">
                <Form.Label>LCA-1 (Practical Performance)</Form.Label>
                <Form.Control
                  type="number"
                  value={marks.lca1}
                  onChange={(e) => setMarks({ ...marks, lca1: e.target.value })}
                  required
                />
              </Form.Group>
              <Form.Group controlId="lca2" className="mt-3">
                <Form.Label>LCA-2 (Active Learning/Project)</Form.Label>
                <Form.Control
                  type="number"
                  value={marks.lca2}
                  onChange={(e) => setMarks({ ...marks, lca2: e.target.value })}
                  required
                />
              </Form.Group>
              <Form.Group controlId="lca3" className="mt-3">
                <Form.Label>LCA-3 (End term practical/oral)</Form.Label>
                <Form.Control
                  type="number"
                  value={marks.lca3}
                  onChange={(e) => setMarks({ ...marks, lca3: e.target.value })}
                  required
                />
              </Form.Group>
              <Form.Group controlId="co1" className="mt-3">
                <Form.Label>CO1 (Analyze and apply data preparation techniques)</Form.Label>
                <Form.Control
                  type="number"
                  value={marks.co1}
                  onChange={(e) => setMarks({ ...marks, co1: e.target.value })}
                  required
                />
              </Form.Group>
              <Form.Group controlId="co2" className="mt-3">
                <Form.Label>CO2 (Compare supervised learning algorithms)</Form.Label>
                <Form.Control
                  type="number"
                  value={marks.co2}
                  onChange={(e) => setMarks({ ...marks, co2: e.target.value })}
                  required
                />
              </Form.Group>
              <Form.Group controlId="co3" className="mt-3">
                <Form.Label>CO3 (Compare unsupervised/semi-supervised algorithms)</Form.Label>
                <Form.Control
                  type="number"
                  value={marks.co3}
                  onChange={(e) => setMarks({ ...marks, co3: e.target.value })}
                  required
                />
              </Form.Group>
              <Form.Group controlId="co4" className="mt-3">
                <Form.Label>CO4 (Design ML techniques for real-time applications)</Form.Label>
                <Form.Control
                  type="number"
                  value={marks.co4}
                  onChange={(e) => setMarks({ ...marks, co4: e.target.value })}
                  required
                />
              </Form.Group>
              <Button variant="primary" type="submit" className="mt-4">
                Add Marks
              </Button>
            </Form>
            <h5 className="mt-4">Performance Graph</h5>
            <StudentGraph prn={selectedStudent.prn} />
            <h5 className="mt-4">Send Feedback</h5>
            <FeedbackForm prn={selectedStudent.prn} email={selectedStudent.email} />
          </Col>
        )}
      </Row>
    </Container>
  );
};

export default TeacherDashboard;