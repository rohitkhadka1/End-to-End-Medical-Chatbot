# Deployment Documentation

## Docker Deployment

### Quick Deploy
```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f chatbot
```

### Production Deploy
```bash
# Set production environment
export FLASK_ENV=production

# Deploy with scaling
docker-compose up -d --scale chatbot=3
```

## Monitoring & Health

### Health Check
```bash
curl http://localhost:5000/health
```

### Logs
```bash
# Application logs
tail -f logs/chatbot.log

# Docker logs
docker-compose logs -f
```

## Configuration

### Environment Variables
| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | | OpenRouter API key for LLM |
| `PINECONE_API_KEY` | | Pinecone vector database key |
| `HUGGINGFACEHUB_API_TOKEN` | | HuggingFace model access |
| `GROQ_API_KEY` | | Groq Whisper for voice (optional) |
| `ELEVEN_API_KEY` | | ElevenLabs premium voice (optional) |

### Performance Tuning
- **Rate Limiting**: Adjust `MAX_REQUESTS_PER_MINUTE`
- **Chunk Size**: Optimize embeddings chunk size in `embeddings.py`
- **Model Caching**: Enable model instance reuse
- **Redis**: Use for session management and caching

## Security

### Production Checklist
- [ ] Change `SECRET_KEY` in production
- [ ] Set `FLASK_DEBUG=False`
- [ ] Configure proper CORS origins
- [ ] Enable rate limiting
- [ ] Use HTTPS in production
- [ ] Secure API keys in environment variables
- [ ] Regular security updates

### Rate Limiting
```python
# Default limits
- 30 requests/minute per IP
- 200 requests/day per IP
- Voice endpoints: Lower limits