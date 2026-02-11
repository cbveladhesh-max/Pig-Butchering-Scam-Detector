# Design Document

## Overview

This document describes the technical design for the Pig Butchering Scam Detector browser extension. The system consists of a lightweight browser extension (client) and a cloud-based AI analysis service (backend) that work together to detect and prevent pig butchering scams in real-time.

## Architecture

### High-Level Architecture (MVP)

```
┌─────────────────────────────────────────────────────────┐
│                     Browser                              │
│  ┌────────────────────────────────────────────────────┐ │
│  │        Browser Extension (Client)                   │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │ │
│  │  │   Content    │  │  Background  │  │  Popup   │ │ │
│  │  │   Script     │  │   Service    │  │    UI    │ │ │
│  │  └──────┬───────┘  └──────┬───────┘  └──────────┘ │ │
│  │         │                  │                        │ │
│  └─────────┼──────────────────┼────────────────────────┘ │
│            │                  │                          │
└────────────┼──────────────────┼──────────────────────────┘
             │                  │
             │                  │ HTTPS/REST API
             │                  ▼
             │         ┌──────────────────┐
             │         │   FastAPI Server │
             │         └────────┬─────────┘
             │                  │
             │         ┌────────▼─────────────────────────┐
             │         │    Backend Service               │
             │         │  ┌─────────────────────────────┐ │
             │         │  │  NLP Scam Intent Model      │ │
             │         │  │  (DistilBERT fine-tuned)    │ │
             │         │  └─────────────────────────────┘ │
             │         │  ┌─────────────────────────────┐ │
             └─────────┼─▶│  Image Authenticity Module  │ │
                       │  │  (CNN Deepfake Classifier)  │ │
                       │  └─────────────────────────────┘ │
                       └──────────────────────────────────┘
```

### Component Responsibilities

#### Browser Extension Components

1. **Content Script**
   - Injected into web pages
   - Extracts visible text content from page
   - Extracts images (profile pictures, content images)
   - Monitors DOM changes for dynamic content
   - Displays warning overlays for high-risk content

2. **Background Service Worker**
   - Manages API communication with backend
   - Implements simple caching (5-minute TTL)
   - Coordinates between content script and popup
   - Manages extension state and storage
   - Updates toolbar badge with risk level

3. **Popup UI**
   - Displays current page risk assessment
   - Shows scam probability and image authenticity scores
   - Lists detected risk indicators
   - Provides basic settings (enable/disable)

#### Backend Components

1. **FastAPI Server**
   - Single REST API endpoint for analysis
   - Request validation
   - Response formatting
   - Basic error handling

2. **NLP Scam Intent Model**
   - Fine-tuned DistilBERT model
   - Analyzes text for scam patterns
   - Outputs scam probability (0-1) and confidence score
   - Identifies key linguistic features

3. **Image Authenticity Module**
   - CNN-based classifier for deepfake detection
   - Analyzes images for AI-generation artifacts
   - Outputs authenticity score (0-1)
   - Returns deepfake likelihood

## Data Models

### Extension Storage Schema

```typescript
interface ExtensionSettings {
  enabled: boolean;
  showWarnings: boolean;
  cacheEnabled: boolean;
  cacheTTL: number; // seconds, default 300 (5 minutes)
}

interface CachedAssessment {
  url: string;
  contentHash: string;
  riskScore: number;
  riskLevel: RiskLevel;
  timestamp: number;
}

interface RiskAssessment {
  url: string;
  riskScore: number; // 0-100
  riskLevel: RiskLevel;
  scamProbability: number; // 0-1
  imageAuthenticity: number; // 0-1
  indicators: string[];
  timestamp: number;
}

enum RiskLevel {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}
```

### API Request/Response Schema

```typescript
// Analysis Request
interface AnalysisRequest {
  url: string;
  text: string;
  images: string[]; // Array of image URLs
}

// Analysis Response
interface AnalysisResponse {
  riskScore: number; // 0-100 combined score
  riskLevel: RiskLevel;
  scamProbability: number; // 0-1 from NLP model
  scamConfidence: number; // 0-1 NLP confidence
  imageAuthenticity: number; // 0-1 average across images
  deepfakeLikelihood: number; // 0-1 average across images
  indicators: string[]; // List of detected risk factors
  recommendations: string[];
}
```

## Component Design

### 1. Content Script (content.js)

**Responsibilities:**
- Extract page content (text and images)
- Send data to background service
- Display warnings based on risk level

**Key Functions:**

```typescript
class ContentExtractor {
  extractText(): string {
    // Extract visible text from page body
    // Limit to first 2000 characters for MVP
  }
  
  extractImages(): string[] {
    // Extract image URLs (profile pics, content images)
    // Limit to first 5 images for MVP
  }
}

class WarningOverlay {
  show(riskLevel: RiskLevel, indicators: string[]): void;
  hide(): void;
}
```

**Implementation Notes:**
- Extract text from document.body.innerText
- Limit text to 2000 characters (performance + API costs)
- Extract images from <img> tags
- Show overlay only for HIGH and CRITICAL risk levels

### 2. Background Service Worker (background.js)

**Responsibilities:**
- Communicate with backend API
- Cache recent assessments
- Update toolbar badge
- Manage extension state

**Key Functions:**

```typescript
class APIClient {
  async analyzeContent(request: AnalysisRequest): Promise<AnalysisResponse> {
    // POST to /api/analyze
    // Handle errors with try-catch
    // Return default safe response on failure
  }
}

class CacheManager {
  get(contentHash: string): CachedAssessment | null;
  set(contentHash: string, assessment: CachedAssessment): void;
  // Simple in-memory cache with 5-minute TTL
}

class BadgeManager {
  updateBadge(tabId: number, riskLevel: RiskLevel): void {
    // Set badge color: green/yellow/orange/red
    // Set badge text: L/M/H/C
  }
}
```

**Implementation Notes:**
- Use fetch() for API calls
- Simple in-memory cache (Map with timestamps)
- Hash content using simple string hash (not crypto)
- No retry logic for MVP (fail fast)

### 3. Popup UI (popup.html + popup.js)

**Responsibilities:**
- Display current page assessment
- Show risk scores and indicators
- Provide enable/disable toggle

**UI Components:**

```typescript
class RiskDisplay {
  render(assessment: RiskAssessment): void {
    // Show risk level badge
    // Display scam probability
    // Display image authenticity
    // List indicators
  }
}
```

**UI Layout:**
- Risk level badge (colored)
- Scam probability: X%
- Image authenticity: Y%
- Detected indicators (bullet list)
- Enable/disable toggle

### 4. NLP Scam Intent Model (Backend)

**Model Architecture:**
- Base: DistilBERT (distilbert-base-uncased)
- Fine-tuning: Binary classification head
- Input: Text (max 512 tokens)
- Output: Scam probability + confidence

**Training Approach (Simplified for MVP):**
- Use pre-trained DistilBERT
- Fine-tune on small labeled dataset (100-500 examples)
- Scam examples: Investment promises, urgency, romance + money
- Non-scam examples: Normal conversations, legitimate content

**Key Patterns to Detect:**
- Investment/crypto keywords: "guaranteed returns", "passive income", "trading platform"
- Urgency: "limited time", "act now", "don't miss out"
- Trust-building + money: "I care about you" + "investment opportunity"
- Platform redirection: "use this app", "download here"

**Implementation:**

```python
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

class ScamIntentModel:
    def __init__(self):
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=2
        )
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        # Load fine-tuned weights if available
    
    def predict(self, text: str) -> dict:
        inputs = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            scam_prob = probs[0][1].item()
            confidence = max(probs[0]).item()
        
        return {
            'scam_probability': scam_prob,
            'confidence': confidence
        }
```

### 5. Image Authenticity Module (Backend)

**Model Architecture:**
- CNN-based binary classifier (ResNet18 or EfficientNet-B0)
- Input: Images (resized to 224x224)
- Output: Authenticity score (0-1)

**Training Approach (Simplified for MVP):**
- Use pre-trained CNN (ImageNet weights)
- Fine-tune on deepfake dataset (e.g., subset of FaceForensics++)
- Real images: Authentic photos
- Fake images: AI-generated, deepfakes

**Implementation:**

```python
from torchvision import models, transforms
import torch
from PIL import Image
import requests
from io import BytesIO

class ImageAuthenticityDetector:
    def __init__(self):
        self.model = models.resnet18(pretrained=True)
        # Replace final layer for binary classification
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)
        # Load fine-tuned weights if available
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def analyze(self, image_url: str) -> dict:
        try:
            response = requests.get(image_url, timeout=5)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
            input_tensor = self.transform(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                authenticity = probs[0][0].item()  # Real class
                deepfake_likelihood = probs[0][1].item()  # Fake class
            
            return {
                'authenticity_score': authenticity,
                'deepfake_likelihood': deepfake_likelihood
            }
        except Exception as e:
            # Return neutral scores on error
            return {
                'authenticity_score': 0.5,
                'deepfake_likelihood': 0.5
            }
```

### 6. FastAPI Backend (main.py)

**Responsibilities:**
- Single endpoint for content analysis
- Load and run both models
- Combine scores into final risk assessment

**Implementation:**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Load models on startup
nlp_model = ScamIntentModel()
image_model = ImageAuthenticityDetector()

class AnalysisRequest(BaseModel):
    url: str
    text: str
    images: List[str]

class AnalysisResponse(BaseModel):
    riskScore: int
    riskLevel: str
    scamProbability: float
    scamConfidence: float
    imageAuthenticity: float
    deepfakeLikelihood: float
    indicators: List[str]
    recommendations: List[str]

@app.post("/api/analyze")
async def analyze_content(request: AnalysisRequest) -> AnalysisResponse:
    # Run NLP model
    nlp_result = nlp_model.predict(request.text)
    
    # Run image model on all images
    image_results = []
    for img_url in request.images[:5]:  # Limit to 5 images
        result = image_model.analyze(img_url)
        image_results.append(result)
    
    # Average image scores
    avg_authenticity = sum(r['authenticity_score'] for r in image_results) / len(image_results) if image_results else 1.0
    avg_deepfake = sum(r['deepfake_likelihood'] for r in image_results) / len(image_results) if image_results else 0.0
    
    # Combine scores (simple weighted average)
    # 60% NLP, 40% Image
    combined_score = (nlp_result['scam_probability'] * 0.6 + avg_deepfake * 0.4) * 100
    
    # Determine risk level
    if combined_score < 25:
        risk_level = 'low'
    elif combined_score < 50:
        risk_level = 'medium'
    elif combined_score < 75:
        risk_level = 'high'
    else:
        risk_level = 'critical'
    
    # Generate indicators
    indicators = []
    if nlp_result['scam_probability'] > 0.6:
        indicators.append("Suspicious language patterns detected")
    if avg_deepfake > 0.6:
        indicators.append("Images may be AI-generated or manipulated")
    
    # Generate recommendations
    recommendations = []
    if risk_level in ['high', 'critical']:
        recommendations.append("Exercise extreme caution")
        recommendations.append("Verify identity through video call")
        recommendations.append("Do not send money or personal information")
    
    return AnalysisResponse(
        riskScore=int(combined_score),
        riskLevel=risk_level,
        scamProbability=nlp_result['scam_probability'],
        scamConfidence=nlp_result['confidence'],
        imageAuthenticity=avg_authenticity,
        deepfakeLikelihood=avg_deepfake,
        indicators=indicators,
        recommendations=recommendations
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

## API Design

### Endpoints

#### POST /api/analyze
Analyzes content for scam indicators.

**Request:**
```json
{
  "url": "https://example.com",
  "text": "I have a great investment opportunity for you...",
  "images": [
    "https://example.com/profile.jpg",
    "https://example.com/photo.jpg"
  ]
}
```

**Response:**
```json
{
  "riskScore": 78,
  "riskLevel": "high",
  "scamProbability": 0.82,
  "scamConfidence": 0.89,
  "imageAuthenticity": 0.35,
  "deepfakeLikelihood": 0.65,
  "indicators": [
    "Suspicious language patterns detected",
    "Images may be AI-generated or manipulated"
  ],
  "recommendations": [
    "Exercise extreme caution",
    "Verify identity through video call",
    "Do not send money or personal information"
  ]
}
```

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

## Security Design

### Extension Security

1. **Content Security Policy (CSP)**
   - Restrict script sources to extension only
   - No inline scripts
   - HTTPS-only API connections

2. **Permissions**
   - Minimal permissions: activeTab, storage
   - No broad host permissions

3. **Data Handling**
   - No permanent storage of analyzed content
   - Simple hash for cache keys (not cryptographic)

### API Security

1. **Transport Security**
   - TLS 1.3 for all connections
   - HTTPS only

2. **Input Validation**
   - Request size limits (text: 2000 chars, images: 5 max)
   - URL validation
   - Content sanitization

3. **Privacy**
   - No logging of user content
   - Ephemeral processing (no data retention)

## Performance Optimization

### Extension Performance

1. **Content Extraction**
   - Limit text to 2000 characters
   - Limit images to 5 per page
   - No DOM observation for MVP (manual trigger only)

2. **Caching**
   - Simple in-memory cache
   - 5-minute TTL
   - Max 50 cached entries

3. **Resource Usage**
   - Lazy initialization
   - Minimal background processing

### Backend Performance

1. **Model Optimization**
   - Use DistilBERT (smaller, faster than BERT)
   - Batch size 1 for MVP (single request processing)
   - CPU inference acceptable for demo

2. **Response Time**
   - Target: <3 seconds end-to-end
   - NLP inference: ~1s
   - Image inference: ~1s per image
   - Network overhead: ~500ms

## Testing Strategy

### Unit Tests
- Test content extraction functions
- Test cache management
- Test score calculation logic
- Target: 70% code coverage for MVP

### Integration Tests
- Test API endpoint with sample data
- Test extension-backend communication
- Test model inference with known inputs

### Property-Based Tests
- Test risk score calculation properties
- Test cache behavior properties

### Manual Testing
- Test extension on real websites
- Test with known scam examples
- Test across Chrome, Firefox, Edge

## Deployment

### Extension Deployment
1. Build extension bundle with Vite
2. Test locally with developer mode
3. Submit to Chrome Web Store (initial target)
4. Firefox and Edge in Phase 2

### Backend Deployment
1. Containerize FastAPI app with Docker
2. Deploy to cloud platform (Heroku/Railway/Render for MVP)
3. Use free tier for demo
4. Environment variables for configuration

## Monitoring and Logging

### Metrics (Minimal for MVP)
- API request count
- Average response time
- Error rate

### Logging
- Error logs only (no content logging)
- Basic request metadata (timestamp, status code)

### Alerts
- API downtime notification

## Future Enhancements (Phase 2)

### Crypto Wallet Risk Analysis
- Integrate blockchain data APIs
- Check wallet addresses against known scam databases
- Analyze transaction patterns
- Graph-based fraud detection

### Advanced Features
- Reverse image search and similarity matching
- Multi-language support
- Real-time conversation monitoring
- Voice/video deepfake detection
- Community-driven scam reporting
- Browser history analysis for pattern detection
- Integration with law enforcement databases

### Scalability Improvements
- Redis caching layer
- Load balancing
- Auto-scaling infrastructure
- Model quantization and optimization
- Batch inference processing

### Enhanced ML Models
- Larger training datasets
- Multi-modal fusion models
- Active learning from user feedback
- Adversarial robustness improvements

## Correctness Properties

### Property 1: Risk Score Bounds
**Description:** For any valid input, the risk score must be in the range [0, 100].

**Rationale:** Risk scores represent percentages and must be bounded to provide consistent user experience.

**Test Strategy:** Generate random valid inputs, verify all risk scores satisfy 0 ≤ score ≤ 100.

### Property 2: Risk Level Consistency
**Description:** Risk level assignment must be consistent with risk score thresholds: Low (0-25), Medium (26-50), High (51-75), Critical (76-100).

**Rationale:** Risk levels provide user-friendly categorization and must accurately reflect underlying scores.

**Test Strategy:** Generate inputs with known risk scores across all ranges, verify correct risk level assignment.

### Property 3: Cache Consistency
**Description:** For identical content within cache TTL, the extension must return the same risk assessment without making additional API calls.

**Rationale:** Caching improves performance and reduces API costs. Cached results must be consistent.

**Test Strategy:** Submit identical content twice within TTL window, verify second request uses cache and returns identical assessment.

### Property 4: Score Combination Monotonicity
**Description:** If both NLP scam probability and image deepfake likelihood increase, the combined risk score must not decrease.

**Rationale:** The scoring system should be monotonic with respect to individual model outputs.

**Test Strategy:** Generate pairs of model outputs where both scores increase, verify combined score is non-decreasing.

### Property 5: Response Time Bound
**Description:** For any valid request, the system must respond within 5 seconds.

**Rationale:** User experience requires timely feedback. Long delays reduce effectiveness.

**Test Strategy:** Submit various content sizes, measure end-to-end latency, verify all responses within bound.

## Testing Framework

**Framework:** Vitest with fast-check for property-based testing

**Rationale:** Vitest provides fast, modern testing with excellent TypeScript support. Fast-check enables property-based testing for verifying correctness properties across many generated inputs.

## Dependencies and Technology Stack

### Browser Extension
- **Language:** TypeScript
- **Build Tool:** Vite
- **Testing:** Vitest + fast-check
- **UI:** Vanilla JS/HTML/CSS (lightweight)

### Backend
- **Language:** Python 3.10+
- **Framework:** FastAPI
- **ML Libraries:** 
  - PyTorch
  - Transformers (Hugging Face)
  - torchvision
  - Pillow (image processing)
- **HTTP Client:** requests
- **Deployment:** Docker

### Infrastructure (MVP)
- **Hosting:** Heroku/Railway/Render (free tier)
- **No database required for MVP**
- **No Redis/caching layer for MVP**

## Acceptance Criteria Mapping

This design addresses the core acceptance criteria from the requirements document:

- **AC-1 (Installation):** Manifest v3 extension with minimal permissions
- **AC-2 (Content Analysis):** ContentExtractor component for text and images
- **AC-3 (AI Assessment):** NLP + Image models with combined scoring
- **AC-4 (User Alerts):** Badge system + warning overlay for high risk
- **AC-5 (Privacy):** No content storage, HTTPS-only, ephemeral processing
- **AC-6 (Performance):** Simple caching, <3s API response target

## MVP Scope Summary

**Included in MVP:**
- Browser extension (Chrome initially)
- Content extraction (text + images)
- NLP scam detection model
- Image authenticity detection model
- Simple risk scoring (weighted average)
- Basic caching (in-memory, 5 min TTL)
- Popup UI with risk display
- Warning overlay for high risk
- FastAPI backend
- Single /api/analyze endpoint

**Deferred to Phase 2:**
- Crypto wallet risk analysis
- Reverse image search
- Multi-browser support (Firefox, Edge)
- Advanced caching (Redis)
- Rate limiting
- User authentication
- Whitelist/blacklist management
- DOM observation for dynamic content
- Multi-language support

## Conclusion

This MVP design provides a focused, achievable solution for a hackathon demo. The two-model approach (NLP + Image) provides meaningful scam detection capabilities while keeping complexity manageable. The architecture is simple, with clear separation between extension and backend, making it easy to develop, test, and demonstrate.
