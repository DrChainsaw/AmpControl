package ampcontrol.model.training.data.processing;

import ampcontrol.audio.processing.Pipe;
import ampcontrol.audio.processing.ProcessingResult;
import ampcontrol.audio.processing.Spectrogram;
import ampcontrol.audio.processing.UnitMaxZeroMean;
import ampcontrol.model.training.data.AudioDataProvider.AudioProcessorBuilder;
import ampcontrol.model.training.data.state.SimpleStateFactory;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Builder for {@link AudioFileProcessor}.
 *
 * @author Christian Skärby
 */
public class AudioFileProcessorBuilder implements AudioProcessorBuilder {
	
	private final List<Path> fileList = new ArrayList<>();
	private Supplier<ProcessingResult.Factory> audioPostProcSupplier = () -> new Pipe(new Spectrogram(512, 32), new UnitMaxZeroMean());
	private Function<Path, AudioSamplingInfo> samplingInfoMapper = new WindowedConsecutiveSamplingInfo(1000, 100, new SimpleStateFactory(0));
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
	
	public AudioFileProcessorBuilder setPostProcSupplier(Supplier<ProcessingResult.Factory> audioPostProcSupplier) {
		this.audioPostProcSupplier = audioPostProcSupplier; return this;
	}
	
	public AudioFileProcessorBuilder setSamplingInfoMapper(Function<Path, AudioSamplingInfo> samplingInfoMapper) {
		this.samplingInfoMapper = samplingInfoMapper; return this;
	}

	public  AudioFileProcessorBuilder setFileSupplierFactory(Function<List<Path>, Supplier<Path>> fileSupplierFactory) {
		this.fileSupplierFactory = fileSupplierFactory; return this;
	}

}
