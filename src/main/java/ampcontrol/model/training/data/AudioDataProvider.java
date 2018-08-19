package ampcontrol.model.training.data;

import ampcontrol.model.training.data.processing.AudioFileProcessorBuilder;
import ampcontrol.model.training.data.processing.AudioProcessor;
import ampcontrol.model.training.data.processing.WindowedConsecutiveSamplingInfo;
import ampcontrol.model.training.data.state.SimpleStateFactory;
import ampcontrol.model.visualize.PlotSpectrogram;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Provides audio {@link TrainingData} from a list of {@link Path Paths} to files with the data to process. Assumes that
 * each file resides in a directory which has the same name as the label attached to it. One
 * {@link AudioProcessorBuilder} is provided for each label to create for different {@link AudioProcessor} instances
 * possibly hiding different implementations for each label.
 *
 * @author Christian Skärby
 */
public class AudioDataProvider implements DataProvider {
	
	private final Map<String, AudioProcessor> processors = new LinkedHashMap<>();
	private final Supplier<String> labelSupplier;
	
	public interface AudioProcessorBuilder {
		 AudioProcessor build();
		 default AudioProcessorBuilder add(Path file) {return this;}
		
	}

	/**
	 * Constructor
	 * @param files list of {@link Path Paths} to files with the data to process. It is assumed that
	 * each file resides in a directory which has the same name as the label attached to it
	 * @param labels Labels and the associated {@link AudioProcessorBuilder}
	 * @param labelSupplier Supplies labels when generating {@link TrainingData}
	 */
	AudioDataProvider(
			List<Path> files,
			Map<String, AudioProcessorBuilder> labels,
			Supplier<String> labelSupplier) {
		
		this.labelSupplier = labelSupplier;
		createProcessors(files, labels);
	}
	
	private void createProcessors(List<Path> files,  Map<String, AudioProcessorBuilder> labels) {
		for(Path file: files) {
			labels.get(file.getParent().getFileName().toString()).add(file);
		}
		
		labels.forEach((key, value) -> processors.put(key, value.build()));
	}
	
	@Override
	public Stream<TrainingData> generateData() {
		return Stream.generate(labelSupplier)
				.map(label -> new TrainingData(label, processors.get(label).getResult()));
	}
	
	public static void main(String[] args) {
		
		List<Path> files = Stream.of(
				Paths.get("E:\\Software projects\\python\\lead_rythm\\data\\lead\\sawsmashedface_56_nohash_13.wav"),
				Paths.get("E:\\Software projects\\python\\lead_rythm\\data\\rythm\\260Disciples left_33_nohash_22.wav")				
				).collect(Collectors.toList());
			
		Map<String, AudioProcessorBuilder> labels = new LinkedHashMap<>();
		labels.put("rythm", new AudioFileProcessorBuilder().setSamplingInfoMapper(new WindowedConsecutiveSamplingInfo(1000, 100, new SimpleStateFactory(666))));
		labels.put("lead", new AudioFileProcessorBuilder());
		AudioDataProvider splitter = new AudioDataProvider(files, labels, () -> "lead");
		
		splitter.generateData().limit(10).forEach(tData -> PlotSpectrogram.plot(tData.result().stream()
				.findFirst()
				.orElseThrow(() -> new RuntimeException("No output!"))));
		
	}

}
