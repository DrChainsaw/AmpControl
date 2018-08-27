package ampcontrol.model.training.data;

import ampcontrol.audio.processing.ProcessingResult;
import ampcontrol.model.training.data.AudioDataProvider.AudioProcessorBuilder;
import ampcontrol.model.training.data.processing.AudioFileProcessorBuilder;
import ampcontrol.model.training.data.processing.SequentialHoldFileSupplier;
import ampcontrol.model.training.data.processing.WindowedConsecutiveSamplingInfo;
import ampcontrol.model.training.data.state.StateFactory;

import java.nio.file.Path;
import java.util.*;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * {@link DataProviderBuilder} for test set used in evaluation. Tries to ensure that all data is "covered" with as
 * few examples as possible.
 *
 * @author Christian Skärby
 */
public class EvalDataProviderBuilder implements DataProviderBuilder {

	private final List<Path> files = new ArrayList<>();
	private final LinkedHashMap<String, AudioProcessorBuilder> labelToBuilder;
	private final Function<Collection<String>, List<String>> labelExpander;
	private final int clipLengthMs;
	private final int windowSizeMs;
	private final Supplier<ProcessingResult.Factory> audioPostProcSupplier;
	private StateFactory stateFactory;

	public EvalDataProviderBuilder(
			Map<String, AudioProcessorBuilder> labelToBuilder,
			Function<Collection<String>, List<String>> labelExpander,
			int clipLengthMs,
			int windowSizeMs,
			Supplier<ProcessingResult.Factory> audioPostProcSupplier,
			StateFactory stateFactory) {
		this.labelToBuilder = new LinkedHashMap<>(labelToBuilder);
		this.labelExpander = labelExpander;
		this.clipLengthMs = clipLengthMs;
		this.windowSizeMs = windowSizeMs;
		this.audioPostProcSupplier = audioPostProcSupplier;
		this.stateFactory = stateFactory;
	}

	@Override
	public AudioDataProvider createProvider() {
		return new AudioDataProvider(
				files,
				labelToBuilder,
				new RandomLabelSupplier<>(labelExpander.apply(labelToBuilder.keySet()), stateFactory.createNewRandom()));
	}

	@Override
	public void addLabel(String label) {
		labelToBuilder.put(label, createAudioFileProcessorBuilder());	
	}


	@Override
	public DataProviderBuilder addFile(Path file) {
		files.add(file); return this;
	}
	
	@Override
	public int getNrofFiles() {
		return files.size();
	}

	private AudioProcessorBuilder createAudioFileProcessorBuilder() {
		return new AudioFileProcessorBuilder()
		.setSamplingInfoMapper(new WindowedConsecutiveSamplingInfo(clipLengthMs, windowSizeMs, stateFactory))
		.setFileSupplierFactory(fileList -> new SequentialHoldFileSupplier(fileList, clipLengthMs / windowSizeMs, stateFactory))
		.setPostProcSupplier(audioPostProcSupplier);
	}
}
