#include "ClothRenderer.cuh"

ClothRenderer::ClothRenderer()
{

}

ClothRenderer::ClothRenderer (uint numVertex, uint numFace)
{
    InitBO(numVertex, numFace);
    _shader = new Shader("shader\\shadow");
    _depthShader = new Shader("shader\\depth");
}

ClothRenderer::~ClothRenderer()
{
    deleteBO(&posVbo, &cuda_posVbo);
    deleteBO(&normVbo, &cuda_normVbo);
    deleteBO(&faceIbo, &cuda_faceIbo);
}

void ClothRenderer::InitBO(uint numVertex, uint numFace)
{
    createBO(&posVbo, &cuda_posVbo, numVertex * sizeof(REAL3), GL_ARRAY_BUFFER);
    createBO(&normVbo, &cuda_normVbo, numVertex * sizeof(REAL3), GL_ARRAY_BUFFER);
    createBO(&faceIbo, &cuda_faceIbo, numFace * sizeof(uint3), GL_ELEMENT_ARRAY_BUFFER);

    cudaGraphicsGLRegisterBuffer(&cuda_posVbo, posVbo, cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&cuda_normVbo, normVbo, cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&cuda_faceIbo, faceIbo, cudaGraphicsMapFlagsWriteDiscard);
}

void ClothRenderer::DrawBO(uint numFace, const Camera& camera)
{
    //UpdateShader(camera);
    RenderScene(numFace);
}

void ClothRenderer::MappingBO(REAL3* pos, REAL3* norm, uint3* fIdx, uint numVertex, uint numFace)
{
    cudaGraphicsMapResources(1, &cuda_posVbo, 0);
    cudaGraphicsMapResources(1, &cuda_normVbo, 0);
    cudaGraphicsMapResources(1, &cuda_faceIbo, 0);

    REAL3* posPtr;
    REAL3* normPtr;
    uint3* facePtr;

    size_t posBytes, normBytes, faceBytes;
    cudaGraphicsResourceGetMappedPointer((void**)&posPtr, &posBytes, cuda_posVbo);
    cudaGraphicsResourceGetMappedPointer((void**)&normPtr, &normBytes, cuda_normVbo);
    cudaGraphicsResourceGetMappedPointer((void**)&facePtr, &faceBytes, cuda_faceIbo);

    CopyVecToVBO << <divup(numVertex, 1024), 1024 >> > (posPtr, pos, numVertex);
    CopyVecToVBO << <divup(numVertex, 1024), 1024 >> > (normPtr, norm, numVertex);
    CopyIdxToIBO << <divup(numFace, 1024), 1024 >> > (facePtr, fIdx, numFace);

    UnMappingBO();
}

void ClothRenderer::UnMappingBO()
{
    cudaGraphicsUnmapResources(1, &cuda_posVbo, 0);
    cudaGraphicsUnmapResources(1, &cuda_normVbo, 0);
    cudaGraphicsUnmapResources(1, &cuda_faceIbo, 0);
}

void ClothRenderer::UpdateShader(Shader& shader, const Camera& camera)
{
    shader.bind();
    shader.update(camera);
}

void ClothRenderer::RenderScene(uint numFace)
{
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, posVbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, normVbo);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, faceIbo);
    glDrawElements(GL_TRIANGLES, numFace * 3, GL_UNSIGNED_INT, 0);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
}

void ClothRenderer::createBO(uint* vbo, struct cudaGraphicsResource** cuda_resource, uint size, GLenum type)
{
    glGenBuffers(1, vbo);
    glBindBuffer(type, *vbo);
    glBufferData(type, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(type, 0);

    cudaGraphicsGLRegisterBuffer(cuda_resource, *vbo, cudaGraphicsMapFlagsWriteDiscard);
}

void ClothRenderer::createFBO(uint* fbo, uint* texture)
{
    glGenFramebuffers(1, fbo);

    glGenTextures(1, texture);
    glBindTexture(GL_TEXTURE_2D, *texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, shadowWidth, shadowHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glBindFramebuffer(GL_FRAMEBUFFER, *fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, *texture, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void ClothRenderer::deleteBO(uint* vbo, struct cudaGraphicsResource** cuda_resource)
{
    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);
    cudaGraphicsUnregisterResource(*cuda_resource);

    *vbo = 0;
}