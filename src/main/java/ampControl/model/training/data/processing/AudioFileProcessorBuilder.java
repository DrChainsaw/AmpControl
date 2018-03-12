package ampControl.model.training.data.processing;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.function.Supplier;

import ampControl.audio.processing.Pipe;
import ampControl.audio.processing.ProcessingResult;
import ampControl.audio.processing.Spectrogram;
import ampControl.audio.processing.UnitMaxZeroMean;
import ampControl.model.training.data.AudioDataProvider.AudioProcessorBuilder;

/**
 * Builder for {@link AudioFileProcessor}.
 *
 * @author Christian Skärby
 */
public class AudioFileProcessorBuilder implements AudioProcessorBuilder {
	
	private final List<Path> fileList = new ArrayList<>();
	private Supplier<ProcessingResult.Processing> audioPostProcSupplier = () -> new Pipe(new Spectrogram(512, 32), new UnitMaxZeroMean());
	private Function<Path, AudioSamplingInfo> samplingInfoMapper = new WindowedConsecutiveSamplingInfo(1000, 100);
	private Function<List<Path>, Supplier<Path>> fileSupplierFactory = fileList -> new RandomFileSupplier(new Random(666), fileList);
	
	@Override
	public AudioFileProcessor build() {
		List<Path> copy = new ArrayList<Path>(fileList);
		return new AudioFileProcessor(fileSupplierFactory.apply(copy), samplingInfoMapper, audioPostProcSupplier);
	}

	@Override
	public AudioProcessorBuilder add(Path file) {
		fileList.add(file); return this;
	}
	
	public AudioFileProcessorBuilder setPostProcSupplier(Supplier<ProcessingResult.Processing> audioPostProcSupplier) {
		this.audioPostProcSupplier = audioPostProcSupplier; return this;
	}
	
	public AudioFileProcessorBuilder setSamplingInfoMapper(Function<Path, AudioSamplingInfo> samplingInfoMapper) {
		this.samplingInfoMapper = samplingInfoMapper; return this;
	}

	public  AudioFileProcessorBuilder setFileSupplierFactory(Function<List<Path>, Supplier<Path>> fileSupplierFactory) {
		this.fileSupplierFactory = fileSupplierFactory; return this;
	}

}
