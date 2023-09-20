from abc import ABC, abstractmethod
from typing import Callable, Generic, Sequence, TypeVar

import attrs

T = TypeVar("T")


@attrs.frozen
class ReceivedMessage(Generic[T]):
    """
    Payload of a message recieved from the remote queue along with
    queue-specific callabels that can be used to manage that message.

    :param payload: the payload.
    :param acknowledge_fn: callable that acknowledges the queue
        that the message has been processed.
    :param extend_lease_fn: callable that let's extends the lease of
        the received message.
    """

    payload: T
    acknowledge_fn: Callable = lambda: None
    extend_lease_fn: Callable = lambda _: None
    approx_receive_count: int = 0


class PushMessageQueue(ABC, Generic[T]):
    name: str

    @abstractmethod
    def push(self, payloads: Sequence[T]) -> None:
        ...


class PullMessageQueue(ABC, Generic[T]):
    name: str

    # TODO: add an 'acknowledge all' element to the return value
    #       to handle acks efficiently
    @abstractmethod
    def pull(self, max_num: int = 1) -> list[ReceivedMessage[T]]:
        ...


class MessageQueue(PushMessageQueue[T], PullMessageQueue[T]):
    ...
