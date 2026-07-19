import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", default=".nxml-control")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    import uvicorn

    from nxml_control.api import create_app

    uvicorn.run(create_app(state_dir=args.state_dir), host=args.host, port=args.port)
