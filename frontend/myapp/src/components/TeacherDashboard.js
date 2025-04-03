import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import { 
  Container, 
  Row, 
  Col, 
  Table, 
  Button, 
  Form, 
  Card, 
  Tab, 
  Tabs,
  Badge,
  Alert,
  Spinner,
  Modal,
  Dropdown
} from 'react-bootstrap';
import { 
  BarChart, 
  PersonFill, 
  JournalBookmark, 
  GraphUp, 
  Envelope,
  CheckCircleFill,
  XCircleFill,
  InfoCircleFill,
  BoxArrowRight,
  Fullscreen,
  FullscreenExit
} from 'react-bootstrap-icons';
import StudentGraph from './StudentGraph';
import FeedbackForm from './FeedbackForm';
import './TeacherDashboard.css';

// Add these handler functions




const TeacherDashboard = () => {
  const navigate = useNavigate();
  const [students, setStudents] = useState([]);
  const [selectedStudent, setSelectedStudent] = useState(null);
  const [generatingReport, setGeneratingReport] = useState(false);
  const [dashboardView, setDashboardView] = useState('default'); // 'default', 'compact', 'detailed'

  const dashboardStyles = {
    default: {
      cardPadding: '20px',
      fontSize: '1rem',
      graphHeight: '500px'
    },
    compact: {
      cardPadding: '10px',
      fontSize: '0.9rem',
      graphHeight: '400px'
    },
    detailed: {
      cardPadding: '25px',
      fontSize: '1.1rem',
      graphHeight: '600px'
    }
  };

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

  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [activeTab, setActiveTab] = useState('marks');
  const [showConfirmation, setShowConfirmation] = useState(false);
  const [fullscreenGraph, setFullscreenGraph] = useState(false);
  const [performanceData, setPerformanceData] = useState(null);

  useEffect(() => {
    const checkAuth = () => {
      const token = localStorage.getItem('user');
      if (!token) {
        navigate('/');
      }
    };

    checkAuth();
    
    const fetchStudents = async () => {
      try {
        setLoading(true);
        const response = await axios.get('http://localhost:5000/api/students', {
          headers: {
            Authorization: `Bearer ${localStorage.getItem('token')}`
          }
        });
        setStudents(response.data.students);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching students:', error);
        setError('Failed to load student data. Please try again.');
        setLoading(false);
        if (error.response?.status === 401) {
          handleLogout();
        }
      }
    };
    fetchStudents();
  }, [navigate]);

  const handleLogout = () => {
    // Clear user data
    localStorage.removeItem('user');

    
    // Redirect to login page
    navigate('/');

    
  };
// Add these state variables at the top of your component
const [absentOptions, setAbsentOptions] = useState({
  cca1: false,
  cca2: false,
  cca3: false,
  lca1: false,
  lca2: false,
  lca3: false,
});

const [absentReasons, setAbsentReasons] = useState({
  cca1: '',
  cca2: '',
  cca3: '',
  lca1: '',
  lca2: '',
  lca3: '',
});

const handleAbsentCheckboxChange = (field) => {
  const newAbsentOptions = {
    ...absentOptions,
    [field]: !absentOptions[field]
  };
  
  setAbsentOptions(newAbsentOptions);
  
  if (newAbsentOptions[field]) {
    setMarks({
      ...marks,
      [field]: '0'
    });
  }
  
  if (!newAbsentOptions[field]) {
    setAbsentReasons({
      ...absentReasons,
      [field]: ''
    });
  }
};
const handleAbsentReasonChange = (field, value) => {
  setAbsentReasons({
    ...absentReasons,
    [field]: value
  });
};
const handleAddMarks = async (e) => {
  e.preventDefault();
  try {
    setLoading(true);
    
    // Prepare the data to send
    const marksData = {
      prn: selectedStudent.prn,
      ...marks,
      absentReasons: {}
    };

    // Add absent reasons only for checked fields
    Object.keys(absentOptions).forEach(key => {
      if (absentOptions[key] && absentReasons[key]) {
        marksData.absentReasons[key] = absentReasons[key];
      }
    });

    const response = await axios.post('http://localhost:5000/api/add-marks', 
      marksData,
      {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      }
    );
    
    if (response.data.success) {
      setSuccess('Marks added successfully!');
      // Reset all form states
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
      setAbsentOptions({
        cca1: false,
        cca2: false,
        cca3: false,
        lca1: false,
        lca2: false,
        lca3: false,
      });
      setAbsentReasons({
        cca1: '',
        cca2: '',
        cca3: '',
        lca1: '',
        lca2: '',
        lca3: '',
      });
      setShowConfirmation(false);
    } else {
      setError('Failed to add marks. Please try again.');
    }
  } catch (error) {
    console.error('Error adding marks:', error);
    setError('An error occurred while adding marks.');
    if (error.response?.status === 401) {
      handleLogout();
    }
  } finally {
    setLoading(false);
  }
};

  const handleInputChange = (e) => {
    const { id, value } = e.target;
    if (value === '' || (Number(value) >= 0 && Number(value) <= 100)) {
      setMarks({ ...marks, [id]: value });
    }
  };

  const handleStudentSelect = (student) => {
    setSelectedStudent(student);
    setActiveTab('marks');
    setError(null);
    setSuccess(null);
  };

  const generateStudentReport = async () => {
    if (!selectedStudent) return;
    
    setGeneratingReport(true);
    try {
      const response = await axios.get(
        `http://localhost:5000/api/generate-student-report/${selectedStudent.prn}`,
        {
          responseType: 'blob',
          headers: {
            Authorization: `Bearer ${localStorage.getItem('token')}`
          }
        }
      );
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `Student_Report_${selectedStudent.prn}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error('Error generating report:', error);
      setError('Failed to generate report');
    } finally {
      setGeneratingReport(false);
    }
  };
  
  const generateOverallReport = async () => {
    setGeneratingReport(true);
    try {
      const response = await axios.get(
        'http://localhost:5000/api/generate-overall-report',
        {
          responseType: 'blob',
          headers: {
            Authorization: `Bearer ${localStorage.getItem('token')}`
          }
        }
      );
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'Class_Performance_Report.pdf');
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error('Error generating report:', error);
      setError('Failed to generate report');
    } finally {
      setGeneratingReport(false);
    }
  };
  
  
  const renderMarksForm = () => (
    
    <Card className="shadow-sm mb-4">
      <Card.Header className="bg-primary text-white d-flex align-items-center">
        <JournalBookmark className="me-2" />
        <h5 className="mb-0">Add Assessment Marks</h5>
      </Card.Header>
      <Card.Body>
        {success && <Alert variant="success" onClose={() => setSuccess(null)} dismissible>{success}</Alert>}
        {error && <Alert variant="danger" onClose={() => setError(null)} dismissible>{error}</Alert>}
        
        <Form onSubmit={handleAddMarks}>
        <Row>
          <Col md={6}>
            <h6 className="text-primary mb-3">Continuous Assessments</h6>
            <Form.Group controlId="cca1" className="mb-3">
              <div className="d-flex align-items-center mb-2">
                <Form.Check
                  type="checkbox"
                  label="Student was absent"
                  checked={absentOptions.cca1}
                  onChange={() => handleAbsentCheckboxChange('cca1')}
                  className="me-3"
                />
                {absentOptions.cca1 && (
                  <Form.Select
                    value={absentReasons.cca1}
                    onChange={(e) => handleAbsentReasonChange('cca1', e.target.value)}
                    style={{ width: '250px' }}
                  >
                    <option value="">Select reason...</option>
                    <option value="Absent">Absent</option>
                    <option value="Present but not submitted">Present but not submitted</option>
                    <option value="Failed in exam">Failed in exam</option>
                  </Form.Select>
                )}
              </div>
              <Form.Label>CCA-1 <Badge bg="info">10 marks</Badge></Form.Label>
              <Form.Control
                type="number"
                value={marks.cca1}
                onChange={handleInputChange}
                min="0"
                max="10"
                required
                disabled={absentOptions.cca1}
              />
            </Form.Group>
            
            <Form.Group controlId="cca2" className="mb-3">
              <div className="d-flex align-items-center mb-2">
                <Form.Check
                  type="checkbox"
                  label="Student was absent"
                  checked={absentOptions.cca2}
                  onChange={() => handleAbsentCheckboxChange('cca2')}
                  className="me-3"
                />
                {absentOptions.cca2 && (
                  <Form.Select
                    value={absentReasons.cca2}
                    onChange={(e) => handleAbsentReasonChange('cca2', e.target.value)}
                    style={{ width: '250px' }}
                  >
                    <option value="">Select reason...</option>
                    <option value="Absent">Absent</option>
                    <option value="Present but not submitted">Present but not submitted</option>
                    <option value="Failed in exam">Failed in exam</option>
                  </Form.Select>
                )}
              </div>
              <Form.Label>CCA-2 <Badge bg="info">5 marks</Badge></Form.Label>
              <Form.Control
                type="number"
                value={marks.cca2}
                onChange={handleInputChange}
                min="0"
                max="5"
                required
                disabled={absentOptions.cca2}
              />
            </Form.Group>
            
            <Form.Group controlId="cca3" className="mb-3">
              <div className="d-flex align-items-center mb-2">
                <Form.Check
                  type="checkbox"
                  label="Student was absent"
                  checked={absentOptions.cca3}
                  onChange={() => handleAbsentCheckboxChange('cca3')}
                  className="me-3"
                />
                {absentOptions.cca3 && (
                  <Form.Select
                    value={absentReasons.cca3}
                    onChange={(e) => handleAbsentReasonChange('cca3', e.target.value)}
                    style={{ width: '250px' }}
                  >
                    <option value="">Select reason...</option>
                    <option value="Absent">Absent</option>
                    <option value="Present but not submitted">Present but not submitted</option>
                    <option value="Failed in exam">Failed in exam</option>
                  </Form.Select>
                )}
              </div>
              <Form.Label>CCA-3 (Mid term) <Badge bg="info">15 marks</Badge></Form.Label>
              <Form.Control
                type="number"
                value={marks.cca3}
                onChange={handleInputChange}
                min="0"
                max="15"
                required
                disabled={absentOptions.cca3}
              />
            </Form.Group>
          </Col>
          
          <Col md={6}>
            <h6 className="text-primary mb-3">Lab Assessments</h6>
            <Form.Group controlId="lca1" className="mb-3">
              <div className="d-flex align-items-center mb-2">
                <Form.Check
                  type="checkbox"
                  label="Student was absent"
                  checked={absentOptions.lca1}
                  onChange={() => handleAbsentCheckboxChange('lca1')}
                  className="me-3"
                />
                {absentOptions.lca1 && (
                  <Form.Select
                    value={absentReasons.lca1}
                    onChange={(e) => handleAbsentReasonChange('lca1', e.target.value)}
                    style={{ width: '250px' }}
                  >
                    <option value="">Select reason...</option>
                    <option value="Absent">Absent</option>
                    <option value="Present but not submitted">Present but not submitted</option>
                    <option value="Failed in exam">Failed in exam</option>
                  </Form.Select>
                )}
              </div>
              <Form.Label>LCA-1 (Practical Performance)</Form.Label>
              <Form.Control
                type="number"
                value={marks.lca1}
                onChange={handleInputChange}
                min="0"
                max="100"
                required
                disabled={absentOptions.lca1}
              />
            </Form.Group>
            
            <Form.Group controlId="lca2" className="mb-3">
              <div className="d-flex align-items-center mb-2">
                <Form.Check
                  type="checkbox"
                  label="Student was absent"
                  checked={absentOptions.lca2}
                  onChange={() => handleAbsentCheckboxChange('lca2')}
                  className="me-3"
                />
                {absentOptions.lca2 && (
                  <Form.Select
                    value={absentReasons.lca2}
                    onChange={(e) => handleAbsentReasonChange('lca2', e.target.value)}
                    style={{ width: '250px' }}
                  >
                    <option value="">Select reason...</option>
                    <option value="Absent">Absent</option>
                    <option value="Present but not submitted">Present but not submitted</option>
                    <option value="Failed in exam">Failed in exam</option>
                  </Form.Select>
                )}
              </div>
              <Form.Label>LCA-2 (Active Learning/Project)</Form.Label>
              <Form.Control
                type="number"
                value={marks.lca2}
                onChange={handleInputChange}
                min="0"
                max="100"
                required
                disabled={absentOptions.lca2}
              />
            </Form.Group>
            
            <Form.Group controlId="lca3" className="mb-3">
              <div className="d-flex align-items-center mb-2">
                <Form.Check
                  type="checkbox"
                  label="Student was absent"
                  checked={absentOptions.lca3}
                  onChange={() => handleAbsentCheckboxChange('lca3')}
                  className="me-3"
                />
                {absentOptions.lca3 && (
                  <Form.Select
                    value={absentReasons.lca3}
                    onChange={(e) => handleAbsentReasonChange('lca3', e.target.value)}
                    style={{ width: '250px' }}
                  >
                    <option value="">Select reason...</option>
                    <option value="Absent">Absent</option>
                    <option value="Present but not submitted">Present but not submitted</option>
                    <option value="Failed in exam">Failed in exam</option>
                  </Form.Select>
                )}
              </div>
              <Form.Label>LCA-3 (End term practical/oral)</Form.Label>
              <Form.Control
                type="number"
                value={marks.lca3}
                onChange={handleInputChange}
                min="0"
                max="100"
                required
                disabled={absentOptions.lca3}
              />
            </Form.Group>
          </Col>
        </Row>
          
          <hr className="my-4" />
          
          <h6 className="text-primary mb-3">Course Outcomes</h6>
          <Row>
            <Col md={6}>
              <Form.Group controlId="co1" className="mb-3">
                <Form.Label>CO1: Data Preparation Techniques</Form.Label>
                <Form.Control
                  type="number"
                  value={marks.co1}
                  onChange={handleInputChange}
                  min="0"
                  max="100"
                  required
                />
              </Form.Group>
              <Form.Group controlId="co2" className="mb-3">
                <Form.Label>CO2: Supervised Learning Algorithms</Form.Label>
                <Form.Control
                  type="number"
                  value={marks.co2}
                  onChange={handleInputChange}
                  min="0"
                  max="100"
                  required
                />
              </Form.Group>
            </Col>
            <Col md={6}>
              <Form.Group controlId="co3" className="mb-3">
                <Form.Label>CO3: Unsupervised/Semi-supervised Algorithms</Form.Label>
                <Form.Control
                  type="number"
                  value={marks.co3}
                  onChange={handleInputChange}
                  min="0"
                  max="100"
                  required
                />
              </Form.Group>
              <Form.Group controlId="co4" className="mb-3">
                <Form.Label>CO4: ML for Real-time Applications</Form.Label>
                <Form.Control
                  type="number"
                  value={marks.co4}
                  onChange={handleInputChange}
                  min="0"
                  max="100"
                  required
                />
              </Form.Group>
            </Col>
          </Row>
          
          <div className="d-flex justify-content-end mt-4">
            <Button 
              variant="primary" 
              type="submit" 
              disabled={loading}
              onClick={() => setShowConfirmation(true)}
            >
              {loading ? (
                <>
                  <Spinner as="span" animation="border" size="sm" role="status" aria-hidden="true" />
                  <span className="ms-2">Adding...</span>
                </>
              ) : (
                'Add Marks'
              )}
            </Button>
          </div>
        </Form>
      </Card.Body>
    </Card>
  );

  const renderPerformanceGraph = () => (
    <div className={`performance-graph-container ${fullscreenGraph ? 'fullscreen' : ''}`}>
      <div className="d-flex justify-content-between align-items-center mb-3">
        <h5 className="mb-0">
          <GraphUp className="me-2" />
          Performance Analysis
        </h5>
        <Button 
          variant="outline-secondary" 
          size="sm"
          onClick={() => setFullscreenGraph(!fullscreenGraph)}
          aria-label={fullscreenGraph ? 'Exit fullscreen' : 'View fullscreen'}
        >
          {fullscreenGraph ? <FullscreenExit /> : <Fullscreen />}
          {fullscreenGraph ? ' Exit Fullscreen' : ' Fullscreen'}
        </Button>
      </div>
      
      <Card className="shadow-sm graph-card">
        <Card.Body className="p-0">
          <div className="graph-wrapper" style={{ height: fullscreenGraph ? '75vh' : '500px' }}>
            <StudentGraph prn={selectedStudent.prn} />
          </div>
        </Card.Body>
      </Card>SHAP Analysis 
      
    </div>
  );
  const renderCOVisualization = () => (
    <Card className="shadow-sm mb-4">
      <Card.Header className="bg-success text-white d-flex align-items-center">
        <GraphUp className="me-2" />
        <h5 className="mb-0">Course Outcomes Visualization</h5>
      </Card.Header>
      <Card.Body>
        {performanceData ? (
          <div className="row">
            {Object.keys(performanceData.coScores).map((co, index) => (
              <Col md={6} lg={4} key={index} className="mb-3">
                <Card className="shadow-sm">
                  <Card.Body>
                    <h6 className="text-primary">{co.toUpperCase()}</h6>
                    <div className="progress" style={{ height: '20px' }}>
                      <div
                        className="progress-bar bg-info"
                        role="progressbar"
                        style={{ width: `${performanceData.coScores[co]}%` }}
                        aria-valuenow={performanceData.coScores[co]}
                        aria-valuemin="0"
                        aria-valuemax="100"
                      >
                        {performanceData.coScores[co]}%
                      </div>
                    </div>
                  </Card.Body>
                </Card>
              </Col>
            ))}
          </div>
        ) : (
          <div className="text-center">
            <Spinner animation="border" variant="primary" />
            <p className="mt-2">Loading CO Visualization...</p>
          </div>
        )}
      </Card.Body>
    </Card>
  );

  const renderSHAPChart = () => (
    <Card className="shadow-sm mb-4">
      <Card.Header className="bg-warning text-white d-flex align-items-center">
        <GraphUp className="me-2" />
        <h5 className="mb-0">SHAP Analysis</h5>
      </Card.Header>
      <Card.Body>
        {performanceData?.shapValues ? (
          <div className="shap-chart-container">
            <StudentGraph data={performanceData.shapValues} type="shap" />
          </div>
        ) : (
          <div className="text-center">
            <Spinner animation="border" variant="primary" />
            <p className="mt-2">Loading SHAP Analysis...</p>
          </div>
        )}
      </Card.Body>
    </Card>
  );

  return (

      <><div>
      {renderCOVisualization()}
      {renderSHAPChart()}
    </div><Container fluid className="teacher-dashboard px-4 py-4" style={{
      fontSize: dashboardStyles[dashboardView].fontSize
    }}>
        <Row className="mb-4 align-items-center">
          <Col>
            <h2 className="fw-bold text-primary">
              <PersonFill className="me-2" />
              Teacher Dashboard
            </h2>
            <p className="text-muted">Manage student assessments and performance</p>
          </Col>
          <Col xs="auto">
            <Dropdown className="ms-2">
              <Dropdown.Toggle variant="outline-secondary" id="dashboard-view-dropdown">
                View: {dashboardView.charAt(0).toUpperCase() + dashboardView.slice(1)}
              </Dropdown.Toggle>
              <Dropdown.Menu>
                <Dropdown.Item onClick={() => setDashboardView('default')}>Default View</Dropdown.Item>
                <Dropdown.Item onClick={() => setDashboardView('compact')}>CO'yS View</Dropdown.Item>
                <Dropdown.Item onClick={() => setDashboardView('detailed')}>Detailed View</Dropdown.Item>
              </Dropdown.Menu>
            </Dropdown>
          </Col>
          <Col xs="auto" className="d-flex gap-2">
            <Button
              variant="outline-primary"
              onClick={generateStudentReport}
              disabled={!selectedStudent || generatingReport}
            >
              {generatingReport ? (
                <Spinner as="span" animation="border" size="sm" />
              ) : (
                'Student Report'
              )}
            </Button>

            <Button
              variant="outline-danger"
              onClick={handleLogout}
            >
              <BoxArrowRight className="me-1" />
              Logout
            </Button>
          </Col>
        </Row>

        <Row>
          <Col md={5} lg={4} xl={3}>
            <Card className="shadow-sm h-100">
              <Card.Header className="bg-primary text-white">
                <h5 className="mb-0">Student Roster</h5>
              </Card.Header>
              <Card.Body className="p-0">
                {loading && !selectedStudent ? (
                  <div className="text-center py-5">
                    <Spinner animation="border" variant="primary" />
                  </div>
                ) : error ? (
                  <Alert variant="danger" className="m-3">
                    {error}
                  </Alert>
                ) : (
                  <div className="student-list-container">
                    <Table hover className="mb-0">
                      <thead className="bg-light">
                        <tr>
                          <th>PRN</th>
                          <th>Name</th>
                          <th>Action</th>
                        </tr>
                      </thead>
                      <tbody>
                        {students.map((student) => (
                          <tr
                            key={student.prn}
                            className={selectedStudent?.prn === student.prn ? 'table-active' : ''}
                          >
                            <td>{student.prn}</td>
                            <td>{student.name}</td>
                            <td>
                              <Button
                                variant={selectedStudent?.prn === student.prn ? 'primary' : 'outline-primary'}
                                size="sm"
                                onClick={() => handleStudentSelect(student)}
                              >
                                {selectedStudent?.prn === student.prn ? 'Selected' : 'Select'}
                              </Button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </Table>
                  </div>
                )}
              </Card.Body>
            </Card>
          </Col>

          <Col md={7} lg={8} xl={9}>
            {selectedStudent ? (
              <Card className="shadow-sm">
                <Card.Header className="bg-light">
                  <div className="d-flex justify-content-between align-items-center">
                    <div>
                      <h4 className="mb-0">
                        {selectedStudent.name}
                        <small className="text-muted ms-2">(PRN: {selectedStudent.prn})</small>
                      </h4>
                      <div className="text-muted small">{selectedStudent.email}</div>
                    </div>
                    <Badge bg="info" className="fs-6">
                      {selectedStudent.department || 'Computer Science'}
                    </Badge>
                  </div>
                </Card.Header>

                <Card.Body>
                  <Tabs
                    activeKey={activeTab}
                    onSelect={(k) => setActiveTab(k)}
                    className="mb-4"
                  >
                    <Tab eventKey="marks" title={<><JournalBookmark className="me-1" /> Assessments</>}>
                      {renderMarksForm()}
                    </Tab>
                    <Tab eventKey="performance" title={<><GraphUp className="me-1" /> Performance</>}>
                      {renderPerformanceGraph()}
                    </Tab>
                    <Tab eventKey="feedback" title={<><Envelope className="me-1" /> Feedback</>}>
                      <Card className="shadow-sm">
                        <Card.Header className="bg-primary text-white d-flex align-items-center">
                          <Envelope className="me-2" />
                          <h5 className="mb-0">Send Feedback</h5>
                        </Card.Header>
                        <Card.Body>
                          <FeedbackForm
                            prn={selectedStudent.prn}
                            email={selectedStudent.email} />
                        </Card.Body>
                      </Card>
                    </Tab>
                  </Tabs>
                </Card.Body>
              </Card>
            ) : (
              <Card className="shadow-sm text-center py-5">
                <InfoCircleFill size={48} className="text-muted mb-3" />
                <h4>No Student Selected</h4>
                <p className="text-muted">Please select a student from the list to view details</p>
              </Card>
            )}
          </Col>
        </Row>

        {/* Confirmation Modal */}
        <Modal show={showConfirmation} onHide={() => setShowConfirmation(false)}>
          <Modal.Header closeButton>
            <Modal.Title>Confirm Marks Submission</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <p>Are you sure you want to submit these marks for {selectedStudent?.name}?</p>
            <div className="alert alert-warning">
              <strong>Note:</strong> Once submitted, marks cannot be edited without admin approval.
            </div>
          </Modal.Body>
          <Modal.Footer>
            <Button variant="secondary" onClick={() => setShowConfirmation(false)}>
              Cancel
            </Button>
            <Button variant="primary" onClick={handleAddMarks} disabled={loading}>
              {loading ? 'Submitting...' : 'Confirm Submission'}
            </Button>
          </Modal.Footer>
        </Modal>
      </Container>
    </>
  );
}

export default TeacherDashboard;