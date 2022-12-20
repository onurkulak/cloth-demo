// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <vector>

// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>
GLFWwindow* window;

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtc/reciprocal.hpp>
#include "glm/gtx/string_cast.hpp"
#include <glm/gtc/epsilon.hpp>
#include <algorithm>    // std::sort
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <omp.h>

using namespace glm;
using namespace std;

#include <iostream>
#include <common/shader.hpp>
#include <common/texture.hpp>
#include <common/controls.hpp>
// #include "PQP.h"


#include <eigen3/Eigen/Dense>


float M_EPSILON = 0.0001f;
bool verbose = false;
const uint backupBufferSize=6;

const int objCnt = 2;
const int dynamicObjCnt = objCnt-1;
const int maxObjCount = 100;
const float attachmentConstraintWeight=10000.0f;


vector<glm::vec3> getClothVertices(vector<glm::vec3> & normals, float clothLength, float quadDensity, vector<unsigned int> & indices, vector<glm::vec2> & uv){
	// convention is 3rd point is always the one with the right angle
	

	float quadLen = 1.0f/quadDensity;
	int quadCnt = round(clothLength*quadDensity);
	
	int pointCnt=quadCnt+1;

	vector<vec3> vertices;
	for(int i=0; i<=quadCnt; i++){
		for(int j=0; j<=quadCnt; j++){
			vec3 offset;
			offset.x = -clothLength*0.5 + quadLen * i;
			offset.y = -clothLength*0.5 + quadLen * j;

			normals.push_back(vec3(0.0,0.0,-1.0));
			vertices.push_back(offset);
			uv.push_back(vec2(1.0-i/float(quadCnt), j/float(quadCnt)));

			if(i<quadCnt && j<quadCnt){
				int this_index = (i*pointCnt) + j;
				indices.push_back(this_index+pointCnt+1);
				indices.push_back(this_index);
				indices.push_back(this_index+1);
				indices.push_back(this_index);
				indices.push_back(this_index+pointCnt+1);
				indices.push_back(this_index+pointCnt);
			}
		}
	}
/*
	for(int i=0; i<vertices.size(); i++)
		cout << to_string (vertices[i]) << endl;
	for(int i=0; i<normals.size(); i++)
		cout << to_string (normals[i]) << endl;

	for(int i=0; i<indices.size(); i++)
		cout << to_string (indices[i]) << endl;
*/
	assert(l1Norm(normals[0], normalize(cross(vertices[indices[1]] - vertices[indices[0]], vertices[indices[2]] - vertices[indices[0]]))) < M_EPSILON);
	assert(l1Norm(normals[0], normalize(cross(vertices[indices[4]] - vertices[indices[3]], vertices[indices[5]] - vertices[indices[3]]))) < M_EPSILON);

	return vertices;
}


// return the triangularized vertices for a unit cube
vector<glm::vec3> getRoomVertices(vector<glm::vec3> & normals, float length){
	
	// discarding the top section

	GLfloat g_vertex_buffer_data[] = {
    -1.0f,-1.0f,-1.0f, // triangle 1 : begin
    -1.0f,-1.0f, 1.0f,
    -1.0f, 1.0f, 1.0f, // triangle 1 : end
    1.0f, 1.0f,-1.0f, // triangle 2 : begin
    -1.0f,-1.0f,-1.0f,
    -1.0f, 1.0f,-1.0f, // triangle 2 : end
    1.0f,-1.0f, 1.0f,
    -1.0f,-1.0f,-1.0f,
    1.0f,-1.0f,-1.0f,
    1.0f, 1.0f,-1.0f,
    1.0f,-1.0f,-1.0f,
    -1.0f,-1.0f,-1.0f,
    -1.0f,-1.0f,-1.0f,
    -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f,-1.0f,
    1.0f,-1.0f, 1.0f,
    -1.0f,-1.0f, 1.0f,
    -1.0f,-1.0f,-1.0f,
    -1.0f, 1.0f, 1.0f,
    -1.0f,-1.0f, 1.0f,
    1.0f,-1.0f, 1.0f,
    1.0f, 1.0f, 1.0f,
    1.0f,-1.0f,-1.0f,
    1.0f, 1.0f,-1.0f,
    1.0f,-1.0f,-1.0f,
    1.0f, 1.0f, 1.0f,
    1.0f,-1.0f, 1.0f,
    1.0f, 1.0f, 1.0f,
    1.0f, 1.0f,-1.0f,
    -1.0f, 1.0f,-1.0f,
    1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f,-1.0f,
    -1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, 1.0f,
    1.0f,-1.0f, 1.0f
	};


	vector<glm::vec3> triangleVertices;
	for(int i = 0; i < 36*3; i+=3){
		triangleVertices.push_back(glm::vec3(g_vertex_buffer_data[i], g_vertex_buffer_data[i+1], g_vertex_buffer_data[i+2]));
	}
	
	for (int i = 0; i < triangleVertices.size(); i+=3){
		
		glm::vec3 normal(0.0f, 0.0f, 0.0f);
		for (int j = 0; j < 3 ; j++){
			normal += triangleVertices[i+j];
			triangleVertices[i+j] *= length;
		}

		for (int j = 0; j < 3; j++){
			if( abs(normal [j]) > 2.5f){
				normal[j] = (normal[j] < 0) - (normal[j] > 0);
			}
			else{
				normal[j] = 0.0f;
			}
		}

		for (int j=0; j<3; j++){
			normals.push_back(normal);
		}
		// cout << normal.x << " " << normal.y << " " << normal.z << endl;

		if (glm::dot(normal, glm::cross(triangleVertices[i+1] - triangleVertices[i],
							triangleVertices[i+1] - triangleVertices[i+2])) > 0){
								swap(triangleVertices[i+1], triangleVertices[i+2]);
							}
	}
	
	return triangleVertices;
}


void performUnitTests(){

	vector<unsigned int> vboIndex;

	vboIndex.push_back(3);
	vboIndex.push_back(0);
	vboIndex.push_back(1);
	vboIndex.push_back(0);
	vboIndex.push_back(3);
	vboIndex.push_back(2);

	Eigen::Matrix3f A;

	A.setIdentity();
	A.array() -= 1.0/4.0f;
	Eigen::Matrix3f A2;
	A2.array() = A.array().pow(2.0f);

	Eigen::Matrix<float, 4, 6> S;
	S.setZero();

	for(int c=0; c<vboIndex.size(); c++){
		S(vboIndex[c], c) = 1.0f;
	}

	
	Eigen::Matrix<float, 4, 4> Y0;
	Y0.setZero();

	Eigen::Matrix<float, 4, 4> Y1;
	Y1.setZero();

	for(int c=0; c<2; c++){
		Eigen::Array3i tempIndex{vboIndex[c*3], vboIndex[c*3+1], vboIndex[c*3+2]};
		Y0(tempIndex, tempIndex) += A.transpose()*A;
	}

	for(int c=0; c<2; c++){
		Y1 += S.block<4,3>(0,c*3) * (A.transpose()*A) * S.block<4,3>(0,c*3).transpose();
	}

	if((Y0-Y1).norm() >0.0f){
		cout << Y0 << endl;
		cout << Y1 << endl;
	}
	assert((Y0-Y1).norm()==0.0f);
}

int projectiveDynamics(Eigen::Matrix<float, -1, 3, Eigen::RowMajor> q[maxObjCount], Eigen::Matrix<float, -1, 3> pC[maxObjCount], Eigen::Matrix<float, -1, 3> moMat[maxObjCount],
	Eigen::Matrix3f A[maxObjCount], Eigen::MatrixXf ST[maxObjCount], Eigen::LLT<Eigen::MatrixXf> L[maxObjCount], 
	Eigen::Matrix<float, -1, 3> b[maxObjCount], Eigen::Matrix<float, -1, 3, Eigen::RowMajor> normals[maxObjCount], vector<unsigned int> index[dynamicObjCnt], 
	float h, Eigen::Matrix<float, -1, 3> Fext[maxObjCount], float quadLengthConstraint, float particleMass, Eigen::Matrix<float, -1, 3, Eigen::RowMajor> v[maxObjCount],
	 vector<pair<int, Eigen::Vector3f>> attachmentConstraints[dynamicObjCnt])
	
	{
	float centerToCornerDistance = sqrt(2.0/9.0) * quadLengthConstraint;
	
	for( int o=0; o<dynamicObjCnt; o++){
		// cout << "global state:" << endl;
		// cout << q[o] << endl;
		moMat[o] = q[o] * (particleMass/h/h) + Fext[o] + (particleMass/h)*v[o];
		v[o] = q[o];
		for(int iterationStep=0; iterationStep<10; iterationStep++){
			// local iteration
			#pragma omp parallel for default(shared) num_threads( 6 )
			for(int t=0; t<pC[o].rows(); t+=3){
				Eigen::RowVector3f p0 = q[o].row(index[o][t]);
				Eigen::RowVector3f p1 = q[o].row(index[o][t+1]);
				Eigen::RowVector3f p2 = q[o].row(index[o][t+2]);
				Eigen::RowVector3f center = (p0+p1+p2) / 3.0f;

				
				pC[o].row(t+2) = center + ((p2-center).normalized() * centerToCornerDistance);
				Eigen::RowVector3f up = (p2 - p0).cross(p1 - p0).normalized();

				Eigen::RowVector3f baseCenter = (center - pC[o].row(t+2))*0.5f + center;
				Eigen::RowVector3f deviationFrombase = (pC[o].row(t+2) - baseCenter).cross(up);
				pC[o].row(t) = baseCenter + deviationFrombase;
				pC[o].row(t+1) = baseCenter - deviationFrombase;
			}
			/* cout << "local constraint input:" << endl;
			for(int i=0; i<index[o].size(); i++)
			{
				cout << q[o].row(index[o][i]) << endl;
			}
			cout << "local constraint output:" << endl;
			
			cout << pC[o] << endl; */

			// global iteration
			// create the b matrix by iterating over all constraints
			// I don't have the M/h^2 term here because I applied it to momat already
			b[o]=moMat[o];
			Eigen::Matrix3f A2 = A[o].transpose()*A[o];
			for(int t=0; t<pC[o].rows(); t+=3){
				b[o] += ST[o](Eigen::all, Eigen::seqN(t,3)) * (A2*pC[o].block<3,3>(t,0));
			}

			for(int at=0; at<attachmentConstraints[o].size(); at++){
				b[o].row(attachmentConstraints[o][at].first) += attachmentConstraints[o][at].second*attachmentConstraintWeight;
			}

			/* cout << "b term:" << endl;
			cout << b[o] << endl; */
			// solve using precomputed cholesky
			#pragma omp parallel for default(shared) num_threads( 3 )
			for(int colC=0; colC<3; colC++){
				// cout << "hellooo" <<  glfwGetTime() << "\t" << omp_get_thread_num() <<endl;
				q[o].col(colC) = L[o].solve(b[o].col(colC));
			}

			/* cout << "global solution results:" << endl;
			cout << q[o] << endl; */
		}

		//calculate new normals
		normals[o].setZero();
		// calculate for each triangle, sum, normalize
		for(int t=0; t<index[o].size(); t+=3){
			Eigen::RowVector3f p0 = q[o].row(index[o][t]);
			Eigen::RowVector3f p1 = q[o].row(index[o][t+1]);
			Eigen::RowVector3f p2 = q[o].row(index[o][t+2]);

			Eigen::RowVector3f crossp = (p1-p0).cross(p2-p0).normalized();
			normals[o].row(index[o][t]) += crossp;
			normals[o].row(index[o][t+1]) += crossp;
			normals[o].row(index[o][t+2]) += crossp;
		}
		normals[o].rowwise().normalize();
		v[o] = (q[o] - v[o])/h;
	}
	
	return 0;
}


Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> eigenFromVec3(const vector<vec3> & v){
	Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> m(v.size(), 3);
	memcpy(m.data(), &v[0], sizeof(vec3)*v.size());
	return m;
}


int main( int argc, char * argv[] )
{

	float roomRadius = 25.0f;
	// edge length of a square cloth. it will be like a curtain in a windy + gravity area
	float clothLength = 1.0f;
	// how many quads are there for each unit of cloth length
	float quadDensity = 10.0f;
	// quadWeight is distributed among the vertices to make a mass matrix
	float clothWeight = 0.25f;
	// first arg is # of polygons

	int gravityOn=0;
	float breezeStrength=0.1;
	float fixedTimeStep = 1.0f/30.f;


	if(argc > 1){
		clothLength = atof(argv[1]);
		quadDensity = atof(argv[2]);
		gravityOn = atoi(argv[3]);
		breezeStrength = atof(argv[4]);
		clothWeight = atof(argv[5]);
		fixedTimeStep = atof(argv[6]);
	}
	

	// Initialise GLFW
	if( !glfwInit() )
	{
		fprintf( stderr, "Failed to initialize GLFW\n" );
		getchar();
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    int windowWidth = 3800;
    int windowHeight = 2100;

	// Open a window and create its OpenGL context
	window = glfwCreateWindow( windowWidth, windowHeight, "Tutorial 0 - Keyboard and Mouse", NULL, NULL);
	if( window == NULL ){
		fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n" );
		getchar();
		glfwTerminate();
		return -1;
	}
    glfwMakeContextCurrent(window);

	// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		getchar();
		glfwTerminate();
		return -1;
	}

	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    // Hide the mouse and enable unlimited mouvement
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    // Set the mouse at the center of the screen
    glfwPollEvents();
    glfwSetCursorPos(window, windowWidth/2, windowHeight/2);

	// Dark blue background
	glClearColor(1.0f, 1.0f, 1.0f, 0.0f);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS); 

	// Cull triangles which normal is not towards the camera
	// we want to render the other side of the cloth too, so keep it
	// glEnable(GL_CULL_FACE);

	GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

	// perform "unit tests"
	performUnitTests();


	vector<glm::vec3> normals[objCnt];
	vector<glm::vec3> vertices[objCnt];
	vector<glm::vec2> uv[objCnt];
	vector<unsigned int> vboIndex[dynamicObjCnt];


	vertices[0] = getClothVertices(normals[0], clothLength, quadDensity, vboIndex[0], uv[0]);
	
	cout << "going to render\t#" << vertices[0].size() << "\t vertices" << endl;
	cout << "going to render\t#" << vboIndex[0].size()/3.0 << "\t triangles" << endl;

	float particleMass = clothWeight/vertices[0].size();

	float quadLength = 1.0/quadDensity;
	vertices[1] = getRoomVertices(normals[1], roomRadius);

	// last buffer is saved for the future prediction
	vec3 locationsBackup[backupBufferSize+1][dynamicObjCnt];
	quat anglesBackup[backupBufferSize+1][dynamicObjCnt];
	vector<glm::vec3> backUpVertices[backupBufferSize+1][dynamicObjCnt];
	vector<glm::vec3> backUpNormals[backupBufferSize+1][dynamicObjCnt];
	for(int i=0; i<dynamicObjCnt; i++){
		for (int j=0; j<=backupBufferSize; j++){
			backUpVertices[j][i].resize(vertices[i].size());
			backUpNormals[j][i].resize(normals[i].size());
		}
	}
	
	// dynamic objects use these eigen library structs, while the wall is rendered using regular vec3
	Eigen::MatrixXf S[maxObjCount];
	
	// A constraint matrix is always the same
	Eigen::Matrix3f A[maxObjCount];

	Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> q[maxObjCount];
	Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> velocities[maxObjCount];
	Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> eigen_normals[maxObjCount];
	Eigen::Matrix<float, -1, 3> b[maxObjCount];
	Eigen::Matrix<float, -1, 3> moMat[maxObjCount];
	// store # force patterns in the buffer, apply them to the object
	const int fExtBufferSize=20;
	// at every # prames pick a new random force pattern to apply
	const int forceBufferChangePeriod=5;
	Eigen::Matrix<float, -1, 3> Fext[fExtBufferSize][maxObjCount];
	Eigen::LLT<Eigen::MatrixXf> L[maxObjCount];
	// this vetor holds the constraint restricted points
	Eigen::Matrix<float, -1, 3> pC[maxObjCount];

	vector<pair<int, Eigen::Vector3f>> attachmentConstraints[dynamicObjCnt];
	

	cout << "starting physics" << endl;
	double objCreationStartTime = glfwGetTime();
			
	for(int i=0; i<dynamicObjCnt; i++){
		moMat[i] = Eigen::MatrixXf::Zero(vertices[i].size(),3);
		b[i] = Eigen::MatrixXf::Zero(vertices[i].size(),3);
		
		for(int j=0; j<fExtBufferSize; j++){

			Fext[j][i] = Eigen::MatrixXf::Zero(vertices[i].size(),3);
			Eigen::RowVector3f windVector = Eigen::RowVector3f::Random()*(breezeStrength*0.5f);
			// just something temporary to make it cooler in the demo
			windVector.x() = -windVector.x();
			Fext[j][i].rowwise() = windVector;
			
			if (gravityOn) Fext[j][i](Eigen::all,1).array() = -9.8f;

			// make sure there is always a stronger breeze from the z axis
			// Fext[j][i](Eigen::all,1).array() += breezeStrength;
		}
		q[i] = eigenFromVec3(vertices[i]);
		velocities[i] = Eigen::MatrixXf::Zero(vertices[i].size(),3);
		pC[i] = Eigen::MatrixXf(vboIndex[i].size(),3);
		eigen_normals[i] = eigenFromVec3(normals[i]);
		A[i].setIdentity();
		A[i].array() -= 1.0/3.0;
		Eigen::Matrix3f A2 = A[i].transpose()*A[i];
		
		// My version of S is a bit different, it's the transposed version of S in the paper
		// instead of being constraint x vertex count it's verteices x constraints
		cout << "creating selector matrix:\t" << glfwGetTime() - objCreationStartTime << endl;
		S[i] = Eigen::MatrixXf::Zero(vertices[i].size(), vboIndex[i].size());
		for(int c=0; c<vboIndex[i].size(); c++){
			S[i](vboIndex[i][c], c) = 1.0f;
		}

		
		// left part of the Y matrix, particle mass and timestep related
		Eigen::MatrixXf tempY = Eigen::MatrixXf::Identity(vertices[i].size(), vertices[i].size());
		
		tempY *= particleMass / (fixedTimeStep*fixedTimeStep);
		cout << "matrix additions" << endl;
		for(int c=0; c<vboIndex[i].size()/3; c++){
			Eigen::Array3i tempIndex{vboIndex[i][c*3], vboIndex[i][c*3+1], vboIndex[i][c*3+2]};
			tempY(tempIndex, tempIndex) += A2;
		}

		// handle attachment constraints:
		int sideIndices=(int)sqrt(vertices[i].size());
		for (int atIndex=vertices[i].size()-sideIndices; atIndex<vertices[i].size(); atIndex++){
			tempY(atIndex,atIndex) += attachmentConstraintWeight;
			attachmentConstraints[i].push_back(make_pair(atIndex,q[i].row(atIndex)));
		}

		cout << "cholesky start:\t" << glfwGetTime() - objCreationStartTime << endl;
		L[i] = Eigen::LLT<Eigen::MatrixXf>(tempY);
		cout << "cholesky end:\t" << glfwGetTime() - objCreationStartTime << endl;
		Eigen::MatrixXf lltLo = L[i].matrixL();
		cout << "llt error is:\t";
		float LLTError = (tempY-lltLo*lltLo.transpose()).norm();
		cout << LLTError << endl;
		// cout << lltLo << endl;
		// assert(LLTError<M_EPSILON);
	}
	cout << "created auxilary matrices. took:\t" << glfwGetTime() - objCreationStartTime << endl;

	vec3 locations[objCnt];
	quat angles[objCnt];
	for(int i=0; i<objCnt; i++){
		locations[i] = vec3(0.0f);
		angles[i] = quat();
	}

	vec3 specularColor[objCnt];
	vec3 diffuseColor[objCnt];

	specularColor[0] = vec3(0.6, 0.2, 0.5);
	diffuseColor[0] = vec3(0.1);

	specularColor[1] = vec3(0.7);
	diffuseColor[1] = vec3(0.1);
	
	
	GLuint vertexbuffers[objCnt];
	glGenBuffers(objCnt, vertexbuffers);
	GLuint normalbuffers[objCnt];
	glGenBuffers(objCnt, normalbuffers);
	GLuint uvbuffers[objCnt];
	glGenBuffers(objCnt, uvbuffers);


	for(int i=0; i < objCnt; i++){
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffers[i]);
		glBufferData(GL_ARRAY_BUFFER, vertices[i].size() * sizeof(glm::vec3), &vertices[i][0], GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, normalbuffers[i]);
		glBufferData(GL_ARRAY_BUFFER, normals[i].size() * sizeof(glm::vec3), &normals[i][0], GL_STATIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, uvbuffers[i]);
		glBufferData(GL_ARRAY_BUFFER, uv[i].size() * sizeof(glm::vec2), &uv[i][0], GL_STATIC_DRAW);
	}

	GLuint elementBuffers[dynamicObjCnt];
	glGenBuffers(dynamicObjCnt, elementBuffers);
	
	for(int i=0; i < dynamicObjCnt; i++){
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementBuffers[i]);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, vboIndex[i].size() * sizeof(unsigned int), &vboIndex[i][0], GL_STATIC_DRAW);
	}

	GLuint m_shaderProgram = LoadShaders( "GeneralPurposeVertexShader.glsl", "GeneralPurposeFragmentShader.glsl" );
	GLuint edgeDebugProgram = LoadShaders( "EdgeVertex.glsl", "EdgeFragment.glsl" );

	// Get a handle for our "MVP" uniform
	GLuint MatrixID = glGetUniformLocation(m_shaderProgram, "MVP");
	GLuint MatrixMVID = glGetUniformLocation(m_shaderProgram, "MV");
	GLuint diffuseColorID = glGetUniformLocation(m_shaderProgram, "diffuseColor");
	GLuint specularColorID = glGetUniformLocation(m_shaderProgram, "specularColor");
	GLuint TextureID  = glGetUniformLocation(m_shaderProgram, "ts");
	GLuint bRenderImg  = glGetUniformLocation(m_shaderProgram, "renderImg");

	GLuint image[dynamicObjCnt];
	image[0] = loadBMP_custom("./us-md.bmp");


	double preTime = glfwGetTime();
	// Always check that our framebuffer is ok
	if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		return false;

	bool pressable[5] = {true,true,true,true,true};
	double totalTime = 0;
	unsigned long long int frameCntr=0;
	unsigned long long int frozenFrame=0;
	bool firstIteration = true;
 

	setControlsWidthHeight(windowWidth, windowHeight);

	int freezePhysics=0;
	bool renderingCurTime[3]={true,true};
	do{
		double currentTime = glfwGetTime();
		double deltaTime = currentTime-preTime;
		preTime = currentTime;
		if(!firstIteration){
			totalTime+=deltaTime;
			frameCntr++;
		}
		else{
			firstIteration = false;
		}

		if(!freezePhysics){
			memcpy(&locationsBackup[frameCntr%backupBufferSize][0], &locations[0], sizeof(locations));
			memcpy(&anglesBackup[frameCntr%backupBufferSize][0], &angles[0], sizeof(angles));
			for( int i=0; i< dynamicObjCnt; i++){
				memcpy(&backUpVertices[frameCntr%backupBufferSize][i][0], q[i].data(), sizeof(vec3)*vertices[i].size());
				memcpy(&backUpNormals[frameCntr%backupBufferSize][i][0], eigen_normals[i].data(), sizeof(vec3)*normals[i].size());
			}
		}

		if(!freezePhysics){
			// cout << "starting physics" << endl;
			double measure = glfwGetTime();
			freezePhysics = projectiveDynamics(q, pC, moMat, A, S, L, b, eigen_normals, vboIndex, fixedTimeStep, 
											Fext[frameCntr%forceBufferChangePeriod], quadLength, particleMass, velocities, attachmentConstraints);
			// cout << "did physics" << endl;
			if(frameCntr%200==0)
			cout << "physics time for iteration:\t" << glfwGetTime() - measure<< endl;
		}


		if(freezePhysics){
			if (glfwGetKey( window, GLFW_KEY_T ) == GLFW_PRESS && pressable[2]){
				renderingCurTime[0] = !renderingCurTime[0];
				pressable[2] = false;
				cout << "is not rendering previous timestep:\t" << renderingCurTime[0] << endl;
			}
			else if (glfwGetKey( window, GLFW_KEY_T ) == GLFW_RELEASE){
				pressable[2] = true;
			}
			
			if (glfwGetKey( window, GLFW_KEY_Y ) == GLFW_PRESS && pressable[3]){
				renderingCurTime[1] = !renderingCurTime[1];
				pressable[3] = false;
				cout << "is not rendering future timestep:\t" << renderingCurTime[1] << endl;
			}
			else if (glfwGetKey( window, GLFW_KEY_Y ) == GLFW_RELEASE){
				pressable[3] = true;
			}

		}
		else{
			if (glfwGetKey( window, GLFW_KEY_T ) == GLFW_PRESS && pressable[2]){
				pressable[2] = false;
				cout << "printing info:\n";
				/*for(int i=0; i<polyhedraCnt;i++){
					cout << to_string(locations[i]) << '\n'; 
					cout << to_string(velocity[i]) << '\n';
					cout << to_string(angles[i]) << '\n';
					cout << to_string(angularVelocity[i]) << '\n';
				}*/
			}
			else if (glfwGetKey( window, GLFW_KEY_T ) == GLFW_RELEASE){
				pressable[2] = true;
			}
		}

			/*
		if (glfwGetKey( window, GLFW_KEY_C ) == GLFW_PRESS && pressable[4]){
			pressable[4] = false;
			freezePhysics=false;
		}
		else{
			freezePhysics = true;
			if (glfwGetKey( window, GLFW_KEY_C ) == GLFW_RELEASE){
				pressable[4] = true;
			}
		}
		*/

		// Render to the screen
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
        // Render on the whole framebuffer, complete from the lower left corner to the upper right
		glViewport(0,0,windowWidth,windowHeight);

		// Clear the screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Compute the MVP matrix from keyboard and mouse input
		computeMatricesFromInputs();
		if(totalTime < 0.5f){
			setCamera(vec3(0.0f,0.0f,-5.0f), vec3(0.0f,0.0f,1.0f));	
		}
		glm::mat4 ProjectionMatrix = getProjectionMatrix();
		glm::mat4 ViewMatrix = getViewMatrix();
		// Use our shader
		glUseProgram(m_shaderProgram);

		// cout << "start rendering" << endl;

		for (int i = 0; i < dynamicObjCnt; i++){

			// Build the model matrix
			glm::mat4 ModelMatrix;
			if(freezePhysics && !(renderingCurTime[0] && renderingCurTime[1])){
				if(!renderingCurTime[0]){
					ModelMatrix = glm::translate(glm::mat4(), locationsBackup[frozenFrame%backupBufferSize][i]) *  mat4_cast(anglesBackup[frozenFrame%backupBufferSize][i]);
				}
				else{
					ModelMatrix = glm::translate(glm::mat4(), locationsBackup[backupBufferSize][i]) *  mat4_cast(anglesBackup[backupBufferSize][i]);
				}
			}
			else{
				ModelMatrix = glm::translate(glm::mat4(), locations[i]) *  mat4_cast(angles[i]);
			}
			
			glm::mat4 MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;
			glm::mat4 MV = ViewMatrix * ModelMatrix;

			// Bind our texture in Texture Unit 0
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, image[i]);
			// Set our "myTextureSampler" sampler to use Texture Unit 0
			glUniform1i(TextureID, 0);


			// Send our transformation to the currently bound shader, 
			// in the "MVP" uniform
			glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);
			glUniformMatrix4fv(MatrixMVID, 1, GL_FALSE, &MV[0][0]);
			glUniform3f(specularColorID, specularColor[i][0], specularColor[i][1], specularColor[i][2]);
			glUniform3f(diffuseColorID, diffuseColor[i][0], diffuseColor[i][1], diffuseColor[i][2]);
			glUniformMatrix4fv(MatrixMVID, 1, GL_FALSE, &MV[0][0]);
			glUniform1f(bRenderImg, 1.0f);
			
			// 1rst attribute buffer : vertices
			glEnableVertexAttribArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, vertexbuffers[i]);
			glBufferSubData(GL_ARRAY_BUFFER, 0, vertices[i].size() * sizeof(vec3), q[i].data());
			glVertexAttribPointer(
				0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
				3,                  // size
				GL_FLOAT,           // type
				GL_FALSE,           // normalized?
				0,                  // stride
				(void*)0            // array buffer offset
			);

			// 2nd attribute buffer : normals
			glEnableVertexAttribArray(1);
			glBindBuffer(GL_ARRAY_BUFFER, normalbuffers[i]);
			glBufferSubData(GL_ARRAY_BUFFER, 0, normals[i].size() * sizeof(vec3), eigen_normals[i].data());
			glVertexAttribPointer(
				1,                                // attribute
				3,                                // size
				GL_FLOAT,                         // type
				GL_FALSE,                         // normalized?
				0,                                // stride
				(void*)0                          // array buffer offset
			);

			glEnableVertexAttribArray(2);
			glBindBuffer(GL_ARRAY_BUFFER, uvbuffers[i]);
			glBufferSubData(GL_ARRAY_BUFFER, 0, uv[i].size() * sizeof(vec2), &uv[i][0]);
			glVertexAttribPointer(
				2,                                // attribute
				2,                                // size
				GL_FLOAT,                         // type
				GL_FALSE,                         // normalized?
				0,                                // stride
				(void*)0                          // array buffer offset
			);

 			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementBuffers[i]);

			// Draw the triangle !
			 glDrawElements(
				GL_TRIANGLES,      // mode
				vboIndex[i].size(),    // count
				GL_UNSIGNED_INT,   // type
				(void*)0           // element array buffer offset
			);

			//sphereVertices.size()); // draw all triangles
			//planeOffsets[renderedPlaneCnt]);

			glDisableVertexAttribArray(0);
			glDisableVertexAttribArray(1);
			glDisableVertexAttribArray(2);
		}

		for (int i = dynamicObjCnt; i < objCnt; i++){

			// Build the model matrix
			glm::mat4 ModelMatrix;
			ModelMatrix = glm::translate(glm::mat4(), locations[i]) *  mat4_cast(angles[i]);
			
			glm::mat4 MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;
			glm::mat4 MV = ViewMatrix * ModelMatrix;


			// Send our transformation to the currently bound shader, 
			// in the "MVP" uniform
			glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);
			glUniformMatrix4fv(MatrixMVID, 1, GL_FALSE, &MV[0][0]);
			glUniform3f(specularColorID, specularColor[i][0], specularColor[i][1], specularColor[i][2]);
			glUniform3f(diffuseColorID, diffuseColor[i][0], diffuseColor[i][1], diffuseColor[i][2]);
			glUniformMatrix4fv(MatrixMVID, 1, GL_FALSE, &MV[0][0]);
			glUniform1f(bRenderImg, -1.0f);
			
			// 1rst attribute buffer : vertices
			glEnableVertexAttribArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, vertexbuffers[i]);
			glBufferSubData(GL_ARRAY_BUFFER, 0, vertices[i].size() * sizeof(vec3), &vertices[i][0]);
			glVertexAttribPointer(
				0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
				3,                  // size
				GL_FLOAT,           // type
				GL_FALSE,           // normalized?
				0,                  // stride
				(void*)0            // array buffer offset
			);

			// 2nd attribute buffer : normals
			glEnableVertexAttribArray(1);
			glBindBuffer(GL_ARRAY_BUFFER, normalbuffers[i]);
			glBufferSubData(GL_ARRAY_BUFFER, 0, normals[i].size() * sizeof(vec3), &normals[i][0]);
			glVertexAttribPointer(
				1,                                // attribute
				3,                                // size
				GL_FLOAT,                         // type
				GL_FALSE,                         // normalized?
				0,                                // stride
				(void*)0                          // array buffer offset
			);

			// Draw the triangle !
			glDrawArrays(GL_TRIANGLES, 0, vertices[i].size()); // draw all triangles
			//sphereVertices.size()); // draw all triangles
			//planeOffsets[renderedPlaneCnt]);

			glDisableVertexAttribArray(0);
			glDisableVertexAttribArray(1);
		}

		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();


	} // Check if the ESC key was pressed or the window was closed
	while( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
		   glfwWindowShouldClose(window) == 0 );

	// Cleanup VBO and shader
	glDeleteBuffers(1, vertexbuffers);
	glDeleteProgram(m_shaderProgram);
	glDeleteVertexArrays(1, &VertexArrayID);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();

	return 0;
}

