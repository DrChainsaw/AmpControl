package ampcontrol.audio.processing;

/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

import org.datavec.audio.dsp.WindowFunction;
import org.jtransforms.fft.DoubleFFT_1D;

import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Computes spectrogram of input. Ripped from {@link org.datavec.audio.extension.Spectrogram}.
 *
 * @author Jacquet Wong
 */
public class Spectrogram implements ProcessingResult.Factory {

    private static final String fftStr = "fft_";
    private static final String ol = "_olf_";
    private static final String numCap = "(\\d*)";
    private static final Pattern parPattern = Pattern.compile(".*" + fftStr + numCap + ol + numCap + ".*");


    private final int fftWindowSize; // number of sample in fft, the value needed to be a number to power of 2
    private final int timeStride; // 1/overlapFactor overlapping, e.g. 1/4=25% overlapping
    private final double[] window;
    private final DoubleFFT_1D fft;

    public Spectrogram(int fftWindowSize, int spectrogramTimeStride) {
        this.fftWindowSize = fftWindowSize;
        this.timeStride = spectrogramTimeStride;

        WindowFunction window = new WindowFunction();
        window.setWindowType("Hamming");
        this.window = window.generate(fftWindowSize);
        this.fft = new DoubleFFT_1D(fftWindowSize);
    }


    public Spectrogram(String parString) {
        Matcher m = parPattern.matcher(parString);
        if (m.matches()) {
            this.fftWindowSize = Integer.parseInt(m.group(1));
            this.timeStride = fftWindowSize / Integer.parseInt(m.group(2));
        } else {
            throw new IllegalArgumentException("Could not create " + this.getClass().getSimpleName() + " from string " + parString + "!");
        }

        fft = new DoubleFFT_1D(fftWindowSize);
        WindowFunction window = new WindowFunction();
        window.setWindowType("Hamming");
        this.window = window.generate(fftWindowSize);
    }

    @Override
    public String name() {
        return nameStatic() + "_" + fftStr + fftWindowSize + ol + (fftWindowSize / timeStride);
    }

    public static String nameStatic() {
        return "spgr";
    }

    @Override
    public ProcessingResult create(ProcessingResult input) {
        return new Result(input);
    }

    private final class Result implements ProcessingResult {

        private final ProcessingResult input;

        public Result(ProcessingResult input) {
            this.input = input;
        }

        @Override
        public Stream<double[][]> stream() {
            return input.stream().map(inputArr -> {
                final int nrofFrames = inputArr.length;
                final int nrofSamplesInFrame = inputArr[0].length;
                double[][] specgram = null;
                for (int i = 0; i < nrofFrames; i++) {
                    double[][] specgramPart = buildSpectrogram(inputArr[i]);
                    if (specgram == null) {
                        specgram = new double[nrofFrames * specgramPart.length][nrofSamplesInFrame];
                    }
                    System.arraycopy(specgramPart, 0, specgram, i * specgramPart.length, specgramPart.length);
                }
                return specgram;
            });
        }

        /**
         * Build spectrogram
         */
        private double[][] buildSpectrogram(double[] amplitudes) {

            final int nrofFrames = (1 + (amplitudes.length - fftWindowSize) / timeStride);

            double[][] signals = new double[nrofFrames][fftWindowSize];
            double[][] specgram = new double[nrofFrames][fftWindowSize / 2];
            for (int f = 0; f < nrofFrames; f++) {
                int startSample = f * timeStride;
                for (int n = 0; n < fftWindowSize; n++) {
                    signals[f][n] = amplitudes[n + startSample] * window[n];
                }
                calculateFftMagnitudes(signals[f], specgram[f]);
            }

            return specgram;
        }

        private void calculateFftMagnitudes(double[] signals, double[] mag) {
            fft.realForward(signals);
            mag[0] = Math.abs(signals[0]);
            // signals[1] is for some reason equal to real part at bin fftWindowSize / 2 -> Useless, discard it
            for (int j = 2; j < fftWindowSize; j += 2) {
                mag[j / 2] = Math.sqrt(signals[j] * signals[j] + signals[j + 1] * signals[j + 1]);
            }
        }
    }

    public static void main(String[] args) {
        Spectrogram spec = new Spectrogram(32, 8);
        double[] testVec = IntStream.range(0, 1234).mapToDouble(i -> i).toArray();
        ProcessingResult res = spec.create(new SingletonDoubleInput(testVec));
        System.out.println(Arrays.deepToString(res.stream().findAny().get()));
    }
}
