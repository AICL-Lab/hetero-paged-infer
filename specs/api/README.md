# API Specifications

This directory contains API interface specifications for the Heterogeneous Inference Engine.

## Purpose

API specifications define the contracts between system components and external interfaces. These serve as the authoritative source for:

- Interface definitions
- Request/response schemas
- Error codes and handling
- Authentication/authorization requirements

## Specification Formats

| Format | Use Case |
|--------|----------|
| OpenAPI 3.0 (YAML) | REST API definitions |
| GraphQL Schema | GraphQL API definitions |
| Protocol Buffers | gRPC service definitions |
| AsyncAPI | Event-driven APIs |

## Current Status

> **Note**: API specifications are planned for future implementation. Currently, the system uses internal Rust trait interfaces.

## Planned Specifications

### Internal API

- `inference-api.yaml` - Core inference request/response API
- `scheduler-api.yaml` - Scheduler control interface
- `metrics-api.yaml` - Monitoring and metrics endpoints

### External API

- `openapi.yaml` - RESTful API for external clients
- `grpc/` - gRPC service definitions for high-performance clients

## Guidelines

### Creating API Specs

1. **Use OpenAPI 3.0+ format** for REST APIs
2. **Include examples** for all request/response bodies
3. **Document all error codes** with clear messages
4. **Version all APIs** using semantic versioning
5. **Reference requirements** from `/specs/product/`

### Example Structure

```yaml
openapi: 3.0.3
info:
  title: Hetero Inference API
  version: 1.0.0
  
paths:
  /v1/inference:
    post:
      summary: Submit inference request
      operationId: submitInference
      tags:
        - Inference
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/InferenceRequest'
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/InferenceResponse'
                
components:
  schemas:
    InferenceRequest:
      type: object
      required:
        - input
      properties:
        input:
          type: string
          description: Input text for inference
        max_tokens:
          type: integer
          default: 100
        temperature:
          type: number
          format: float
          default: 1.0
```

## Related Documents

- [Product Requirements](../product/) - Business requirements for APIs
- [RFCs](../rfc/) - Technical design decisions
- [Testing Specs](../testing/) - API test specifications
