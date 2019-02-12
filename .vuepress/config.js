module.exports = {
	title: 'Learning AI',
	description: 'Notes in learning AI',
	base: '/Learning-AI/',
	themeConfig: {
		repo: 'chuxubank/Learning-AI',
		editLinks: true,
		nav: [{
				text: 'Home',
				link: '/'
			},
			{
				text: 'deeplearning.ai',
				items: [{
						text: 'Neural Networks and Deep Learning',
						link: '/C1-Neural-Networks-and-Deep-Learning/'
					},
					{
						text: 'Improving Deep Neural Networks',
						link: '/Improving Deep Neural Networks/'
					},
					{
						text: 'Structuring Machine Learning Projects',
						link: '/Structuring Machine Learning Projects/'
					},
					{
						text: 'Convolutional Neural Networks',
						link: '/Convolutional Neural Networks/'
					},
					{
						text: 'Sequence Models',
						link: '/Sequence Models/'
					}
				]
			}
		],
		sidebar: {
			'/C1-Neural-Networks-and-Deep-Learning/': [{
					title: 'Week 1 | Introduction to deep learning',
					children: [
						'W1-Introduction-to-deep-learning/',
						'W1-Introduction-to-deep-learning/L1-Welcome-to-the-Deep-Learning-Specialization',
						'W1-Introduction-to-deep-learning/L2-Introduction-to-Deep-Learning',
					]
				},
				{
					title: 'Week 2 | Neural Networks Basics',
					sidebarDepth: 2,
					children: [
						'W2-Neural-Networks-Basics/',
						'W2-Neural-Networks-Basics/L1-Logistic-Regression-as-a-Neural-Network',
						'W2-Neural-Networks-Basics/L2-Python-and-Vectorization'
					]
				},
				'Practice-Questions',
				'Heroes-of-Deep-Learning'
			]
		}
	},
	plugins: [
		'latex',
		'@vuepress/back-to-top'
	],
	markdown: {
		extendMarkdown: md => {
			md.use(require('markdown-it-checkbox'))
		}
	}
}