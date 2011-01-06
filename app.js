// begin ported glsl simplex noise code
var perm = [151,160,137,91,90,15,
            131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
            190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
            88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
            77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
            102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
            135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
            5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
            223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
            129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
            251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
            49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
            138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180];

// Ken Perlin's proposed gradients for 3D noise (grad3[16][3])
var grad3 = [[0,1,1],[0,1,-1],[0,-1,1],[0,-1,-1],
             [1,0,1],[1,0,-1],[-1,0,1],[-1,0,-1],
             [1,1,0],[1,-1,0],[-1,1,0],[-1,-1,0], // 12 cube edges
             [1,0,-1],[-1,0,-1],[0,-1,1],[0,1,1]]; // 4 more to make 16


// coordinates of the midpoints of each of the 32 edges of a tesseract, just like the 3D
// noise gradients are the midpoints of the 12 edges of a cube. (grad4[32][4])
var grad4 = [[0,1,1,1], [0,1,1,-1], [0,1,-1,1], [0,1,-1,-1], // 32 tesseract edges
             [0,-1,1,1], [0,-1,1,-1], [0,-1,-1,1], [0,-1,-1,-1],
             [1,0,1,1], [1,0,1,-1], [1,0,-1,1], [1,0,-1,-1],
             [-1,0,1,1], [-1,0,1,-1], [-1,0,-1,1], [-1,0,-1,-1],
             [1,1,0,1], [1,1,0,-1], [1,-1,0,1], [1,-1,0,-1],
             [-1,1,0,1], [-1,1,0,-1], [-1,-1,0,1], [-1,-1,0,-1],
             [1,1,1,0], [1,1,-1,0], [1,-1,1,0], [1,-1,-1,0],
             [-1,1,1,0], [-1,1,-1,0], [-1,-1,1,0], [-1,-1,-1,0]];

// This is a look-up table to speed up the decision on which simplex we
// are in inside a cube or hypercube "cell" for 3D and 4D simplex noise.
// It is used to avoid complicated nested conditionals in the GLSL code.
// The table is indexed in GLSL with the results of six pair-wise
// comparisons beween the components of the P=(x,y,z,w) coordinates
// within a hypercube cell.
// c1 = x>=y ? 32 : 0;
// c2 = x>=z ? 16 : 0;
// c3 = y>=z ? 8 : 0;
// c4 = x>=w ? 4 : 0;
// c5 = y>=w ? 2 : 0;
// c6 = z>=w ? 1 : 0;
// offsets = simplex[c1+c2+c3+c4+c5+c6];
// o1 = step(160,offsets);
// o2 = step(96,offsets);
// o3 = step(32,offsets);
// (For the 3D case, c4, c5, c6 and o3 are not needed.)
var simplex4 = new Uint8Array([ 0,64,128,192, 0,64,192,128, 0,0,0,0,
                                0,128,192,64, 0,0,0,0, 0,0,0,0, 0,0,0,0, 64,128,192,0,
                                0,128,64,192, 0,0,0,0, 0,192,64,128, 0,192,128,64,
                                0,0,0,0, 0,0,0,0, 0,0,0,0, 64,192,128,0,
                                0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,
                                0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,
                                64,128,0,192, 0,0,0,0, 64,192,0,128, 0,0,0,0,
                                0,0,0,0, 0,0,0,0, 128,192,0,64, 128,192,64,0,
                                64,0,128,192, 64,0,192,128, 0,0,0,0, 0,0,0,0,
                                0,0,0,0, 128,0,192,64, 0,0,0,0, 128,64,192,0,
                                0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,
                                0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,
                                128,0,64,192, 0,0,0,0, 0,0,0,0, 0,0,0,0,
                                192,0,64,128, 192,0,128,64, 0,0,0,0, 192,64,128,0,
                                128,64,0,192, 0,0,0,0, 0,0,0,0, 0,0,0,0,
                                192,64,0,128, 0,0,0,0, 192,128,0,64, 192,128,64,0]);

// create and load a 2D texture for
// a combined index permutation and gradient lookup table
// this texture is used for 2D and 3D simplex noise
function initPermTexture(gl) {
  var pixels = new Uint8Array(256*256*4);
  for(var i = 0; i < 256; i++) {
    for(var j = 0; j < 256; j++) {
      var offset = (i*256+j)*4;
      var value = perm[(j+perm[i]) & 0xFF];
      pixels[offset]   = grad3[value & 0x0F][0] * 64 + 64; // gradient x
      pixels[offset+1] = grad3[value & 0x0F][1] * 64 + 64; // gradient y
      pixels[offset+2] = grad3[value & 0x0F][2] * 64 + 64; // gradient z
      pixels[offset+3] = value; // permuted index
    }
  }
  gl.activeTexture(gl.TEXTURE1); // switch to texture unit 1
  var texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 256, 0, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.activeTexture(gl.TEXTURE0); // back to texture unit 0
  return texture;
}

function initSimplexTexture(gl) {
  gl.activeTexture(gl.TEXTURE2); // switch to texture unit 2
  var texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 64, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, simplex4);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.activeTexture(gl.TEXTURE0); // and back to texture unit 0
  return texture;
}

function initGradTexture(gl) {
  var pixels = new Uint8Array(256*256*4);
  for(var i = 0; i < 256; i++) {
    for(var j = 0; j < 256; j++) {
      var offset = (i*256+j)*4;
      var value = perm[(j+perm[i]) & 0xFF];
      pixels[offset]   = grad4[value & 0x1F][0] * 64 + 64; // gradient x
      pixels[offset+1] = grad4[value & 0x1F][1] * 64 + 64; // gradient y
      pixels[offset+2] = grad4[value & 0x1F][2] * 64 + 64; // gradient z
      pixels[offset+3] = grad4[value & 0x1F][3] * 64 + 64; // gradient w
    }
  }
  gl.activeTexture(gl.TEXTURE3); // switch to texture unit 3
  var texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 256, 0, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.activeTexture(gl.TEXTURE0); // and back to texture unit 0
  return texture;
}
// end ported glsl noise code

// sets up vertex attribute pointers for given context 'gl'
// and geometry 'geom' as returned from makeBox or makeSquare
// finally renders the geometry with current transformations
var prevGeom = null
function drawGeom(gl, geom) {
  if(prevGeom != geom) {
    // Set up all the vertex attributes for vertices, normals and texCoords
    gl.bindBuffer(gl.ARRAY_BUFFER, geom.vertexObject);
    gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, geom.normalObject);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, geom.texCoordObject);
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 0, 0);

    // Bind the index array
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, geom.indexObject);
    prevGeom = geom;
  }
  // and finally draw the geometry!
  gl.drawElements(gl.TRIANGLES, geom.numIndices, gl.UNSIGNED_BYTE, 0);
  return true;
}

//
// makeSquare
//
// Create a square with vertices, normals and texCoords. Create VBOs for each as well as the index array.
// Return an object with the following properties:
//
//  normalObject        WebGLBuffer object for normals
//  texCoordObject      WebGLBuffer object for texCoords
//  vertexObject        WebGLBuffer object for vertices
//  indexObject         WebGLBuffer object for indices
//  numIndices          The number of indices in the indexObject
//
function makeSquare(ctx)
{
    // plane
    //  v1------v0
    //  |       |
    //  |       |
    //  |       |
    //  v2------v3
    //
    // vertex coords array
    var vertices = new Float32Array(
      [  1, 1, 1,  -1, 1, 1,  -1,-1, 1,   1,-1, 1]    // v0-v1-v2-v3 front
    );
    // normal array
    var normals = new Float32Array(
        [  0, 0, 1,   0, 0, 1,   0, 0, 1,   0, 0, 1]     // v0-v1-v2-v3 front
       );
    // texCoord array
    /*
      var texCoords = new Float32Array(
        [  1, 1,   0, 1,   0, 0,   1, 0]    // v0-v1-v2-v3 front
    );
    */
    var texCoords = new Float32Array(
        [  0.5, 0.5,   -0.5, 0.5,   -0.5, -0.5,   0.5, -0.5]    // v0-v1-v2-v3 front
    );
    // index array
    var indices = new Uint8Array(
        [  0, 1, 2,   0, 2, 3]    // front
    );
    var retval = { };
    retval.normalObject = ctx.createBuffer();
    ctx.bindBuffer(ctx.ARRAY_BUFFER, retval.normalObject);
    ctx.bufferData(ctx.ARRAY_BUFFER, normals, ctx.STATIC_DRAW);

    retval.texCoordObject = ctx.createBuffer();
    ctx.bindBuffer(ctx.ARRAY_BUFFER, retval.texCoordObject);
    ctx.bufferData(ctx.ARRAY_BUFFER, texCoords, ctx.STATIC_DRAW);

    retval.vertexObject = ctx.createBuffer();
    ctx.bindBuffer(ctx.ARRAY_BUFFER, retval.vertexObject);
    ctx.bufferData(ctx.ARRAY_BUFFER, vertices, ctx.STATIC_DRAW);

    ctx.bindBuffer(ctx.ARRAY_BUFFER, null);

    retval.indexObject = ctx.createBuffer();
    ctx.bindBuffer(ctx.ELEMENT_ARRAY_BUFFER, retval.indexObject);
    ctx.bufferData(ctx.ELEMENT_ARRAY_BUFFER, indices, ctx.STATIC_DRAW);
    ctx.bindBuffer(ctx.ELEMENT_ARRAY_BUFFER, null);

    retval.numIndices = indices.length;
    return retval;
}


function init()
{
  // Initialize
  var gl = initWebGL(
    // The id of the Canvas Element
    "example",
    // The ids of the vertex and fragment shaders
    "vshader", "fshader",
    // The vertex attribute names used by the shaders.
    // The order they appear here corresponds to their index
    // used later.
    [ "vNormal", "vColor", "vPosition"],
    // The clear color and depth values
    [ 0.5, 0.5, 0.5, 1 ], 10000);

  // Set some uniform variables for the shaders
  gl.uniform3f(gl.getUniformLocation(gl.program, "lightDir"), 0, 0, 1);

  gl.uniform1i(gl.getUniformLocation(gl.program, "colorSampler"), 0);
  gl.uniform1i(gl.getUniformLocation(gl.program, "permTexture"), 1);
  gl.uniform1i(gl.getUniformLocation(gl.program, "simplexTexture"), 2);
  gl.uniform1i(gl.getUniformLocation(gl.program, "gradTexture"), 3);

  gl.uniform3f(gl.getUniformLocation(gl.program, "sphere1"), 0, -0.25, -1.0);
  gl.uniform3f(gl.getUniformLocation(gl.program, "sphere2"), 0,  0.25, -1.0);
  gl.uniform3f(gl.getUniformLocation(gl.program, "sphere3"), 0.25, 0.0, -1.0);
  gl.uniform1f(gl.getUniformLocation(gl.program, "threshold"), 25.0);
  gl.uniform1f(gl.getUniformLocation(gl.program, "time"), 0.0); // dawn of time

  // Enable texturing
  gl.enable(gl.TEXTURE_2D);

  // Create a box. On return 'gl' contains a 'box' property with
  // the BufferObjects containing the arrays for vertices,
  // normals, texture coords, and indices.
  gl.box = makeBox(gl);
  gl.square = makeSquare(gl);

  // Create some matrices to use later and save their locations in the shaders
  gl.mvMatrix = new J3DIMatrix4();
  gl.u_normalMatrixLoc = gl.getUniformLocation(gl.program, "u_normalMatrix");
  gl.normalMatrix = new J3DIMatrix4();
  gl.u_modelViewMatrixLoc = gl.getUniformLocation(gl.program, "u_modelViewMatrix");

  gl.u_projMatrixLoc = gl.getUniformLocation(gl.program, "u_projMatrix");

  // setup simplex noise lookup tables
  gl.permTexture = initPermTexture(gl);
  gl.simplexTexture = initSimplexTexture(gl);
  gl.gradTexture = initGradTexture(gl);

  // Enable all of the vertex attribute arrays.
  gl.enableVertexAttribArray(0);
  gl.enableVertexAttribArray(1);
  gl.enableVertexAttribArray(2);
  return gl;
}

width = -1;
height = -1;

function reshape(gl)
{
  var canvas = document.getElementById("example");

  if (canvas.width == width && canvas.height == height)
    return;

  width = canvas.width;
  height = canvas.height;

  // Set the viewport and projection matrix for the scene
  gl.viewport(0, 0, width, height);
  gl.perspectiveMatrix = new J3DIMatrix4();
  gl.perspectiveMatrix.perspective(30, width/height, 1, 10000);
  gl.perspectiveMatrix.lookat(0, 0, 100, 0, 0, 0, 0, 1, 0);
}

function drawPicture(gl, state, t)
{
  // Make sure the canvas is sized correctly.
  reshape(gl);

  // Clear the canvas
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  // Make a model/view matrix.
  gl.mvMatrix.makeIdentity();
  gl.mvMatrix.rotate(20, 1,0,0);
  gl.mvMatrix.setUniform(gl, gl.u_modelViewMatrixLoc, false);

  // Update projection matrix uniform shader variable
  gl.perspectiveMatrix.setUniform(gl, gl.u_projMatrixLoc, false);

  // Construct the normal matrix from the model-view matrix and pass it in
  gl.normalMatrix.load(gl.mvMatrix);
  gl.normalMatrix.invert();
  gl.normalMatrix.transpose();
  gl.normalMatrix.setUniform(gl, gl.u_normalMatrixLoc, false);

  gl.mvMatrix.makeIdentity();
  gl.mvMatrix.translate(0, 0, 0);
  gl.mvMatrix.scale(20,20,20);
  gl.mvMatrix.setUniform(gl, gl.u_modelViewMatrixLoc, false);

  gl.uniform3f(gl.getUniformLocation(gl.program, "sphere1"), 0, Math.sin(t) *  0.3, -1.0);
  gl.uniform3f(gl.getUniformLocation(gl.program, "sphere2"), 0, Math.sin(t*1.0) * -0.3, -1.0);
  gl.uniform3f(gl.getUniformLocation(gl.program, "sphere3"), Math.sin(t*3.0) * -0.3, 0.0, -1.0);
  gl.uniform1f(gl.getUniformLocation(gl.program, "time"), t);
  // draw a single rect
  drawGeom(gl, gl.square);

  // Finish up.
  gl.flush();
  // Show the framerate
  framerate.snapshot();
}

var start = function()
{
  var c = document.getElementById("example");
  var w = Math.floor(window.innerWidth  * 0.9);
  var h = Math.floor(window.innerHeight * 0.9);
  var state = {'points':[], 'time': 0.0};

  c.width = w;
  c.height = h;

  var gl = init();
  framerate = new Framerate("framerate");
  var drawIntervalMs = 33.333;
  var t = 0.0;
  setInterval(function() { drawPicture(gl, state, t += 0.01) }, drawIntervalMs);
  //var stateInterval = setInterval(function() {
  //  jQuery.getJSON("/state/aa", {}, function(data, txtStatus, xhr) { state = data; }); clearInterval(stateInterval); }, 1000.0);
}

$(start());

