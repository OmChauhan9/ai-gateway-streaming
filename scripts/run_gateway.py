"""Runner script to start the AI Gateway server."""

import uvicorn
from config.settings import get_settings


def main():
    settings = get_settings()
    print(f"\n{'='*65}")
    print(f"  Starting {settings.app_name} v{settings.version}")
    print(f"  Listening on: http://{settings.host}:{settings.port}")
    print(f"  Primary Upstream:  {settings.primary_provider_name} ({settings.primary_base_url})")
    print(f"  Fallback Upstream: {settings.fallback_provider_name} ({settings.fallback_base_url})")
    print(f"  Prometheus Scrape: http://{settings.host}:{settings.port}/metrics")
    print(f"  Health Endpoint:   http://{settings.host}:{settings.port}/health")
    print(f"{'='*65}\n")

    uvicorn.run(
        "gateway.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
