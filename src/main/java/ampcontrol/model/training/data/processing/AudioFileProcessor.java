package ampcontrol.model.training.data.processing;

import ampcontrol.audio.processing.*;
import ampcontrol.model.training.data.state.SimpleStateFactory;
import ampcontrol.model.visualize.PlotSpectrogram;
import org.datavec.audio.Wave;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * {@link AudioProcessor} which reads a portion of the audio data from a file and produces a {@link ProcessingResult}.
 * TODO: Add testing either using some small wav-file or by stubbing {@link Wave}.
 *
 * @author Christian Skärby
 */
public class AudioFileProcessor implements AudioProcessor {

    private final Supplier<Path> fileSupplier;
    private final Function<Path, AudioSamplingInfo> samplingInfoMapper;
    private final Supplier<ProcessingResult.Factory> resultSupplier;


    /**
     * Constructor
     *
     * @param fileSupplier Supplies {@link Path Paths} to process
     * @param samplingInfoMapper Maps a {@link Path} to an {@link AudioSamplingInfo}
     * @param resultSupplier Provides {@link ProcessingResult.Factory}
     */
    AudioFileProcessor(
            Supplier<Path> fileSupplier,
            Function<Path, AudioSamplingInfo> samplingInfoMapper,
            Supplier<ProcessingResult.Factory> resultSupplier) {
        this.fileSupplier = fileSupplier;
        this.samplingInfoMapper = samplingInfoMapper;
        this.resultSupplier = resultSupplier;
    }

    @Override
    public ProcessingResult getResult() {
        boolean ok = false;
        while (!ok) {
            final Path next = fileSupplier.get();
            //System.out.println("next file: " + next);
            final AudioSamplingInfo info = samplingInfoMapper.apply(next);
            Wave wav = new Wave(next.toAbsolutePath().toString());
            short[] ampData = wav.getSampleAmplitudes();
            final int sampRate = wav.getWaveHeader().getSampleRate();
            final int start = (int) Math.round(info.getStartTime() * sampRate);
            final int nrofSamps = (int) Math.round(info.getLength() * sampRate);

            double[] ampDataTrim = new double[nrofSamps];
            int offs = Math.max(0, start + nrofSamps - ampData.length);
            if (offs > nrofSamps * 0.75) {
                //System.err.println("Incorrect file found: " + next.toString() + " with size " + ampData.length +" and wanted "+ start + " to " + (start+nrofSamps)+ "! Skipping...");
                continue;
            }

            for (int i = 0; i < ampDataTrim.length - offs; i++) {
                ampDataTrim[i] = ampData[i + start];
            }

            ProcessingResult.Factory factory = resultSupplier.get();
            return factory.create(new SingletonDoubleInput(ampDataTrim));
        }
        return null;
    }


    public static void main(String[] args) {

        final Supplier<Path> pathSupplier = new SequentialHoldFileSupplier(Arrays.asList(
                Paths.get("E:\\Software projects\\python\\lead_rythm\\data\\noise\\Ackord left_149_nohash_2.wav"),
                Paths.get("E:\\Software projects\\python\\lead_rythm\\data\\lead\\sawsmashedface_56_nohash_13.wav"),
                Paths.get("E:\\Software projects\\python\\lead_rythm\\data\\rythm\\temp left_45_nohash_9.wav")
        ), 1, new SimpleStateFactory(0));
        // System.out.println(Arrays.toString(new Wave(wavfile.toAbsolutePath().toString()).getSampleAmplitudes()));
        final AudioSamplingInfo aInfo = new AudioSamplingInfo(0.3, 0.05);

        final ProcessingResult.Factory fac = new Pipe(
                new Spectrogram(256, 16), new Log10());

        final AudioProcessor proc = new AudioFileProcessor(pathSupplier, file -> aInfo, () -> fac);

        for(int i = 0; i < 4; i++) {
            ProcessingResult result = proc.getResult();
            System.out.println("result: " + result);
            result.stream().forEach(data -> {
                System.out.println(": time: " + data.length + " freq: " + data[0].length);
                PlotSpectrogram.plot(data);
                //System.out.println("Sum: " + Stream.of(data).mapToDouble(timeFr -> DoubleStream.of(timeFr).sum()).sum());
            });
        }
    }

}
