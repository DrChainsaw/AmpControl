package ampControl.admin;

import java.time.Duration;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import ampControl.admin.service.classifiction.AudioClassificationService;
import ampControl.admin.service.control.AppControlService;
import ampControl.admin.service.Service;
import org.eclipse.paho.client.mqttv3.MqttException;

import com.beust.jcommander.Parameter;

import ampControl.admin.service.control.MessageToActionMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Runs the program main loop. Design borrowed from Winthing project.
 *
 * Basically listens to messages from an {@link AppControlService} and either starts or stops the
 * {@link AudioClassificationService} or exits the application depending on which messages are received.
 *
 * @author Christian Skärby
 */
public class Engine {

	@Parameter(names = "-mqttExit", description = "Mqtt message contents to exit application")
	private String mqttExitMsg = "exit";

	@Parameter(names = "-mqttActAutoMsg", description = "Mqtt message contents to start auto program change")
	private String mqttActMsg = "activateAutoProgramChange";

	@Parameter(names = "-mqttDectAutoMsg", description = "Mqtt message contents to stop auto program change")
	private String mqttDeactMsg = "deactivateAutoProgramChange";

	private static final Logger log = LoggerFactory.getLogger(Engine.class);
	
	private boolean isInit = false;
	private boolean exit = false;
	private AppControlService appControlService;
    private Service service;

	private final Lock runnningLock = new ReentrantLock();
	private final Condition runningCondition = runnningLock.newCondition();
	private final Duration reconnectInterval = Duration.ofSeconds(5);

	/**
	 * Initialization. Reason for this method instead of a constructor is only because Jcommander must have
	 * an instance to set parameters.
	 *
	 * @param appControlService
	 * @param service
	 * @throws MqttException
	 */
	public void initialize(
			AppControlService appControlService,
			Service service) {

		if (isInit) {
			throw new IllegalStateException("May not reinit!");
		}

		this.service = service;
		this.appControlService = appControlService;
		isInit = true;
	}

	/**
	 * Starts the main loop.
	 */
	public void run() {
		if (!isInit) {
			throw new IllegalStateException("Must initialize before running!");
		}
		exit = false;
		runnningLock.lock();
		try {
			while (!exit) {
				log.info("Starting engine...");
				boolean connected = false;
				try {
					MessageToActionMap messageToActionMap = appControlService.start();
					messageToActionMap.mapMessage(mqttActMsg, () -> new Thread(() -> service.start()).start());
					messageToActionMap.mapMessage(mqttDeactMsg, () -> new Thread(() -> service.stop()).start());
					messageToActionMap.mapMessage(mqttExitMsg, () -> stop());
					messageToActionMap.setConnectionFailedAction(() -> pokeCondition());
					connected = true;
				} catch (final MqttException exception) {
					log.info("Could not connect: " + exception.getMessage() + "!");
				}
				if (connected) {
					try {
						log.info("Application started. Waiting for input.");
						runningCondition.await();
					} catch (final InterruptedException exception) {
						stop();                        
					}
				}

				if(!exit) {
					log.info(
							"Trying to reconnect in "+reconnectInterval.getSeconds() +" seconds..."
							);
					try {
						Thread.sleep(reconnectInterval.toMillis());
					} catch (final InterruptedException exception) {
						stop();
					}
				}
			}
		} finally {
			runnningLock.unlock();  
		}
		try {
			log.info("Engine done! " + exit);
			appControlService.stop();
		}  catch (final MqttException disconnectException) {
			log.info("Could not disconnect.");
			disconnectException.printStackTrace();
		}   
	}

	/**
	 * Stops the application.
	 */
	public void stop() {
		log.info("Stopping engine...");
		service.stop();
		exit = true;
		pokeCondition();
	}
	
	private void pokeCondition() {
		runnningLock.lock();
		try {
			runningCondition.signal();
		} finally {
			runnningLock.unlock();
		}
	}
}

