import { useEffect, useRef } from "react";

function CameraFeed() {

  const videoRef = useRef(null);

  useEffect(() => {

    const startCamera = async () => {

      try {

        const stream =
          await navigator.mediaDevices.getUserMedia({
            video: true
          });

        if (videoRef.current) {

          videoRef.current.srcObject =
            stream;
        }

      } catch (error) {

        console.error(
          "Camera access denied:",
          error
        );
      }
    };

    startCamera();

  }, []);

  return (

    <div className="camera-container">

      <video
        ref={videoRef}
        autoPlay
        playsInline
        className="camera-feed"
      />

    </div>

  );
}

export default CameraFeed;