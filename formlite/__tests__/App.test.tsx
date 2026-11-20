import React from 'react';
import {Alert, DeviceEventEmitter, NativeModules, PermissionsAndroid} from 'react-native';
import Renderer from 'react-test-renderer';
let App: typeof import('../App').default;

jest.mock('../assets/camera-setup.svg', () => () => null);
jest.mock('../assets/live-recognition.svg', () => () => null);


const draft = {
  id: 'fixture', started: 1, duration: 12000, unclassified: 4300,
  exercises: [{id: 'squat', name: 'Squats', duration: 7700}],
};
const state = {
  phase: 'idle', ready: false, elapsed: 0, countdown: 0, label: null,
  message: null, error: null, canFinish: false, draft: null,
};

const native = {
  insets: jest.fn(async () => 'null'), addListener: jest.fn(), removeListeners: jest.fn(), openSetup: jest.fn(), closeSetup: jest.fn(),
  start: jest.fn(), switchCamera: jest.fn(), pause: jest.fn(), resume: jest.fn(), finish: jest.fn(),
  state: jest.fn(async () => JSON.stringify(state)), sessions: jest.fn(async () => JSON.stringify([draft])),
  save: jest.fn(async () => { emit('saved', draft); return JSON.stringify(draft); }),
  discard: jest.fn(() => emit('idle', null)),
};

function emit(phase: string, summary: typeof draft | null) {
  DeviceEventEmitter.emit('FormLiteState', JSON.stringify({...state, phase, draft: summary}));
}

beforeEach(() => {
  NativeModules.FormLiteRecognition = native;
  App = require("../App").default;
  jest.spyOn(PermissionsAndroid, 'check').mockResolvedValue(true);
  jest.clearAllMocks();
});

afterEach(() => jest.restoreAllMocks());

test('reviews the duration table, saves, and opens history', async () => {
  let app!: Renderer.ReactTestRenderer;
  await Renderer.act(async () => {app = Renderer.create(<App />);});
  const press = async (title: string) => {
    const button = app.root.findAllByProps({title}).find(node => typeof node.props.press === 'function');
    expect(button).toBeDefined();
    await Renderer.act(async () => button!.props.press());
  };
  await press('Start recognition');
  expect(native.openSetup).toHaveBeenCalledTimes(1);
  await Renderer.act(async () => emit('summary', draft));
  expect(JSON.stringify(app.toJSON())).toContain('Unclassified time');
  expect(JSON.stringify(app.toJSON())).toContain('00:07');
  expect(native.save).not.toHaveBeenCalled();
  await press('Save session');
  expect(native.save).toHaveBeenCalledTimes(1);
  await press('History');
  expect(native.sessions).toHaveBeenCalledTimes(1);
  expect(JSON.stringify(app.toJSON())).toContain('00:12');
  await Renderer.act(async () => app.unmount());
});

test('discard requires confirmation and does not save', async () => {
  let app!: Renderer.ReactTestRenderer;
  const alert = jest.spyOn(Alert, 'alert').mockImplementation((_title, _message, buttons) => {
    buttons?.find(button => button.text === 'Discard')?.onPress?.();
  });
  await Renderer.act(async () => {app = Renderer.create(<App />);});
  const start = app.root.findAllByProps({title: 'Start recognition'}).find(node => node.props.press)!;
  await Renderer.act(async () => start.props.press());
  await Renderer.act(async () => emit('summary', draft));
  const discard = app.root.findAllByProps({title: 'Discard'}).find(node => node.props.press)!;
  await Renderer.act(async () => discard.props.press());
  expect(alert).toHaveBeenCalledTimes(1);
  expect(native.discard).toHaveBeenCalledTimes(1);
  expect(native.save).not.toHaveBeenCalled();
  await Renderer.act(async () => app.unmount());
});

test('passes native controls no React press event argument', async () => {
  let app!: Renderer.ReactTestRenderer;
  await Renderer.act(async () => {app = Renderer.create(<App />);});
  const start = app.root.findAllByProps({title: 'Start recognition'}).find(node => node.props.press)!;
  await Renderer.act(async () => start.props.press());
  await Renderer.act(async () => {
    DeviceEventEmitter.emit('FormLiteState', JSON.stringify({...state, phase: 'setup', ready: true}));
  });
  const control = app.root.findAllByProps({title: 'Switch camera'}).find(node => node.props.press)!;
  await Renderer.act(async () => control.findAll(node => typeof node.props.onPress === 'function')[0].props.onPress({nativeEvent: {}}));
  expect(native.switchCamera).toHaveBeenCalledWith();
  await Renderer.act(async () => app.unmount());
});

test('shows low confidence as a recognition status', async () => {
  let app!: Renderer.ReactTestRenderer;
  await Renderer.act(async () => {app = Renderer.create(<App />);});
  const start = app.root.findAllByProps({title: 'Start recognition'}).find(node => node.props.press)!;
  await Renderer.act(async () => start.props.press());
  await Renderer.act(async () => {
    DeviceEventEmitter.emit('FormLiteState', JSON.stringify({
      ...state, phase: 'active', ready: true, message: 'Not sure yet',
    }));
  });
  const rendered = JSON.stringify(app.toJSON());
  expect(rendered).toContain('Not sure yet');
  expect(rendered).toContain('RECOGNITION STATUS');
  expect(rendered).not.toContain('DETECTED EXERCISE');
  await Renderer.act(async () => app.unmount());
});
