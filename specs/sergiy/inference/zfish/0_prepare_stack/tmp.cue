#items: [1, 2, 3, 4, 5]
#A: {
	x: _
	y: 1
}
[
	for x in #items {{'x': x} & #A},
]
