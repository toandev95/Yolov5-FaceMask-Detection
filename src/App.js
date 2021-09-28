import React, { useEffect, useState, useRef } from "react";
import { Row, Col, Card, Spin } from "antd";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";
import Bluebird from "bluebird";
import _ from "lodash";

import "./App.css";

const names = ["Không khẩu trang", "Có khẩu trang"];

function App() {
  const ref = useRef();
  const webcamRef = useRef();

  const [webcamWidth, setWebcamWidth] = useState(0);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    setWebcamWidth(ref.current.offsetWidth - 24 * 2);

    (async () => {
      const graphModel = await tf.loadGraphModel("/web_model/model.json");

      const detect = async () => {
        if (webcamRef != null) {
          const canvas = webcamRef.current.getCanvas();

          if (!_.isNil(canvas)) {
            const newCanvas = document.getElementById("overlay");
            const ctx = newCanvas.getContext("2d");

            newCanvas.width = canvas.width;
            newCanvas.height = canvas.height;

            tf.engine().startScope();

            const pixels = tf.browser.fromPixels(canvas);
            const inputs = tf.image
              .resizeBilinear(pixels, [320, 320])
              .div(255.0)
              .expandDims(0);
            const ranks = await graphModel.executeAsync(inputs);

            const [boxes, scores, classes, valid_detections] = ranks;

            for (let i = 0; i < valid_detections.dataSync()[0]; ++i) {
              let [x1, y1, x2, y2] = boxes.dataSync().slice(i * 4, (i + 1) * 4);

              x1 *= canvas.width;
              y1 *= canvas.height;

              x2 *= canvas.width;
              y2 *= canvas.height;

              const width = x2 - x1;
              const height = y2 - y1;

              ctx.lineWidth = 3;
              ctx.strokeStyle = "#ff0000";
              ctx.strokeRect(x1, y1, width, height);
            }

            for (let i = 0; i < valid_detections.dataSync()[0]; ++i) {
              let [x1, y1] = boxes.dataSync().slice(i * 4, (i + 1) * 4);

              x1 *= canvas.width;
              y1 *= canvas.height;

              const label = names[classes.dataSync()[i]];
              const score = scores.dataSync()[i].toFixed(2);

              if (score > 0.3) {
                ctx.font = "14px sans-serif";
                ctx.textBaseline = "top";

                const textWidth = ctx.measureText(`${label}: ${score}`).width;
                const textHeight = parseInt("14px sans-serif", 10);

                x1 = x1 - 2;
                y1 = y1 - textHeight;

                ctx.fillStyle = "#ff0000";
                ctx.fillRect(x1, y1, textWidth, textHeight);

                ctx.fillStyle = "#ffffff";
                ctx.fillText(`${label}: ${score}`, x1, y1);
              }
            }

            pixels.dispose();

            tf.engine().endScope();

            setIsLoading(false);
          }
        }

        await Bluebird.delay(80);

        return detect();
      };

      await detect();
    })();
  }, []);

  return (
    <div className="container">
      <Row justify="center">
        <Col span={12}>
          <div ref={ref}>
            <Card title="Camera Thiết Bị">
              <Spin spinning={isLoading}>
                <div className="wrapper">
                  <Webcam
                    ref={webcamRef}
                    width={webcamWidth}
                    screenshotQuality={0.7}
                    mirrored
                    className="webcam"
                  />
                  <canvas id="overlay" className="overlay" />
                </div>
              </Spin>
            </Card>
          </div>
        </Col>
      </Row>
    </div>
  );
}

export default App;
